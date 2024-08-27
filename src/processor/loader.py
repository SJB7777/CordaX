import os
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import h5py
import hdf5plugin  # pylint: disable=unused-import

from src.config.config import load_config, ExpConfig
from src.config.enums import Hertz


class RawDataLoader(ABC):
    """
    Abstract base class for loading raw data from various sources.

    This class defines the interface for loading raw data using different strategies.
    Subclasses must implement the abstract methods to provide specific implementations.

    Attributes:
        file (str): The path to the raw data file.
    """

    @abstractmethod
    def __init__(self, file: str) -> None:
        """
        Initialize the RawDataLoader with the path to the raw data file.

        Args:
            file (str): The path to the raw data file.
        """

    @abstractmethod
    def get_data(self) -> dict[str, npt.NDArray]:
        """
        Retrieve the raw data as a dictionary of numpy arrays.

        Returns:
            dict[str, npt.NDArray]: A dictionary where keys are data identifiers
            and values are numpy arrays containing the raw data.
        """


class HDF5FileLoader(RawDataLoader):
    """Load hdf5 file and remove unmatching data."""
    def __init__(self, file: str):
        """
        Initializes the HDF5FileLoader by loading
        metadata, images, and qbpm data from the given file.

        Parameters:
        - file (str): Path to the HDF5 file.
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f"No such file: {file}")

        self.file: str = file
        self.config: ExpConfig = load_config()

        metadata: pd.DataFrame = pd.read_hdf(self.file, key='metadata')
        merged_df: pd.DataFrame = self.get_merged_df(metadata)

        self.images: npt.NDArray[np.float32] = np.stack(merged_df['image'].values)
        self.qbpm: npt.NDArray[np.float32] = np.stack(merged_df['qbpm'].values)
        self.pump_state: npt.NDArray[np.bool_] = self.get_pump_mask(merged_df)
        self.delay: npt.NDArray[np.float64] = self.get_delay(merged_df)

        # roi_coord = np.array(
        #     self.metadata[
        #         f'detector_{self.config.param.hutch}_{self.config.param.detector}_parameters.ROI'
        #     ].iloc[0][0]
        # )
        # roi = np.array([
        #     roi_coord[self.config.param.x1],
        #     roi_coord[self.config.param.x2],
        #     roi_coord[self.config.param.y1],
        #     roi_coord[self.config.param.y2]
        # ], dtype=np.int_)
        # self.roi_rect = RoiRectangle().from_tuple(roi)

    def get_merged_df(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Merges image and qbpm data with metadata based on timestamps.

        Parameters:
        - metadata (pd.DataFrame): Metadata DataFrame.

        Returns:
        - pd.DataFrame: Merged DataFrame containing metadata, images, and qbpm data.
        """
        with h5py.File(self.file, "r") as hf:
            if "detector" not in hf:
                raise KeyError(f"Key 'detector' not found in {self.file}")

            image_group = hf[f'detector/{self.config.param.hutch.value}/{self.config.param.detector.value}/image']
            images_ts = np.asarray(image_group["block0_items"], dtype=np.int64)
            images = np.asarray(image_group["block0_values"], dtype=np.float32)

            qbpm_group = hf[f'qbpm/{self.config.param.hutch.value}/qbpm1']
            qbpm_ts = np.asarray(qbpm_group['waveforms.ch1/axis1'], dtype=np.int64)
            qbpm = np.stack(
                [qbpm_group[f'waveforms.ch{i + 1}/block0_values'] for i in range(4)],
                axis=0,
                dtype=np.float32
            ).sum(axis=(0, 2))

        image_df = pd.DataFrame(
            {
                "timestamp": images_ts,
                "image": list(images)
            }
        ).set_index('timestamp')

        qbpm_df = pd.DataFrame(
            {
                "timestamp": qbpm_ts,
                "qbpm": list(qbpm)
            }
        ).set_index('timestamp')

        merged_df = pd.merge(image_df, qbpm_df, left_index=True, right_index=True, how='inner')
        return pd.merge(metadata, merged_df, left_index=True, right_index=True, how='inner')

    def get_delay(self, merged_df: pd.DataFrame) -> Union[np.float64, float]:
        """
        Retrieves the delay value from the merged_df.

        Parameters:
        - merged_df (pd.DataFrame): merged_df DataFrame.

        Returns:
        - Union[np.float64, float]: Delay value or NaN if not found.
        """
        if "th_value" in merged_df:
            return np.asarray(merged_df['th_value'], dtype=np.float64)[0]
        if "delay_value" in merged_df:
            return np.asarray(merged_df['delay_value'], dtype=np.float64)[0]
        return np.nan

    def get_pump_mask(self, merged_df: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """
        Generates a pump status mask based on the configuration settings.

        Parameters:
        - merged_df (pd.DataFrame): Merged DataFrame.

        Returns:
        - npt.NDArray[np.bool_]: Pump status mask.
        """
        if self.config.param.pump_setting is Hertz.ZERO:
            return np.zeros(merged_df.shape[0], dtype=np.bool_)
        return np.asarray(
            merged_df[f'timestamp_info.RATE_{self.config.param.xray.value}_{self.config.param.pump_setting.value}'],
            dtype=np.bool_
            )

    def get_data(self) -> dict[str, npt.NDArray]:
        """
        Retrieves data based on pump status.

        Returns:
        - dict[str, npt.NDArray]: Dictionary containing images and qbpm data for both pump-on and pump-off states.
        """
        data: dict[str, npt.NDArray] = {"delay": self.delay}

        poff_images = self.images[~self.pump_state]
        poff_qbpm = self.qbpm[~self.pump_state]
        pon_images = self.images[self.pump_state]
        pon_qbpm = self.qbpm[self.pump_state]

        poff_images = np.maximum(0, poff_images)
        pon_images = np.maximum(0, pon_images)

        if poff_images.size > 0:
            data["poff"] = poff_images
            data["poff_qbpm"] = poff_qbpm
        if pon_images.size > 0:
            data["pon"] = pon_images
            data["pon_qbpm"] = pon_qbpm
        return data


def get_hdf5_images(file: str, config: ExpConfig) -> npt.NDArray:
    """get images form hdf5"""
    with h5py.File(file, "r") as hf:
        if "detector" not in hf:
            raise KeyError(f"Key 'detector' not found in {file}")

        images = np.asarray(hf[f'detector/{config.param.hutch.value}/{config.param.detector.value}/image/block0_values'])

        return np.maximum(images, 0)


if __name__ == "__main__":
    from src.filesystem import get_run_scan_directory
    import time

    config: ExpConfig = load_config()
    load_dir: str = config.path.load_dir
    file: str = get_run_scan_directory(load_dir, 141, 1, 77)


    start = time.time()
    loader = HDF5FileLoader(file)
    print(f"{time.time() - start} sec")
