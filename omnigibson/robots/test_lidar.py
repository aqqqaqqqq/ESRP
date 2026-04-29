import os

from omnigibson.macros import gm
from omnigibson.robots.test import Test


class TestLidar(Test):
    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/new_lidar/fetch/fetch.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/new_lidar/fetch_descriptor.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/new_lidar/fetch.urdf")

    @property
    def eef_usd_path(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/new_lidar/fetch/fetch_eef.usd")}
