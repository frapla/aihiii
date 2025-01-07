import logging
from typing import Literal

import scipy

LOG: logging.Logger = logging.getLogger(__name__)


class UnitConverter:
    def __init__(
        self,
        unit_mass: Literal["g", "kg", "t"],
        unit_length: Literal["mm", "m"],
        unit_time: Literal["s", "ms"],
    ) -> None:
        """Conversion factors from input unit system to system [mm, ms, kg]

        Args:
            unit_mass (Literal[&quot;g&quot;, &quot;kg&quot;, &quot;t&quot;]): unit of mass
            unit_length (Literal[&quot;mm&quot;, &quot;m&quot;]): unit of length
            unit_time (Literal[&quot;s&quot;, &quot;ms&quot;]): unit of time
        """
        # input unit system
        self.__unit_mass = unit_mass
        self.__unit_time = unit_time
        self.__unit_length = unit_length
        self.__unit_velocity = f"{self.__unit_length}/{self.__unit_time}"
        self.__unit_acceleration = f"{self.__unit_length}/{self.__unit_time}^2"
        self.__unit_force = f"({self.__unit_mass}*{self.__unit_length})/{self.__unit_time}^2"
        self.__unit_moment = f"({self.__unit_mass}*{self.__unit_length}^2)/{self.__unit_time}^2"
        self.__unit_pressure = f"{self.__unit_mass}/({self.__unit_length}*{self.__unit_time}^2)"

        # base conversion factors
        self.__conv_time2ms = {"s": 1000, "ms": 1}[unit_time]
        self.__conv_length2mm = {"m": 1000, "mm": 1}[unit_length]
        self.__conv_mass2kg = {"kg": 1, "g": 0.001, "t": 1000}[unit_mass]

    def dummy(self) -> float:
        return 1

    def time2ms(self) -> float:
        """Conversion factor for time from input unit system to unit [ms]

        Returns:
            float: conversion factor
        """
        conv = self.__conv_time2ms
        LOG.debug("Conversion [%s] -> [ms] = %s", self.__unit_time, conv)
        return conv

    def mass2g(self) -> float:
        """Conversion factor for mass from input unit system to unit  [g]

        Returns:
            float: conversion factor
        """
        conv = self.__conv_mass2kg * 1000
        LOG.debug("Conversion [%s] -> [g] = %s", self.__unit_mass, conv)
        return conv

    def length2mm(self) -> float:
        """Conversion factor for length / displacements from input unit system to unit [mm]

        Returns:
            float: conversion factor
        """
        conv = self.__conv_length2mm
        LOG.debug("Conversion [%s] -> [mm] = %s", self.__unit_length, conv)
        return conv

    def mass2kg(self) -> float:
        """Conversion factor for mass from input unit system to unit  [kg]

        Returns:
            float: conversion factor
        """
        conv = self.__conv_mass2kg
        LOG.debug("Conversion [%s] -> [kg] = %s", self.__unit_mass, conv)
        return conv

    def force2kn(self) -> float:
        """Conversion factor for force from input unit system to unit  [kN]

        Returns:
            float: conversion factor
        """
        conv = self.__conv_mass2kg * self.__conv_length2mm / self.__conv_time2ms**2
        LOG.debug("Conversion [%s] -> [kN] = %s", self.__unit_force, conv)
        return conv

    def moment2nm(self) -> float:
        """Conversion factor for moment from input unit system to unit  [Nm]

        Returns:
            float: conversion factor
        """
        conv = (self.__conv_mass2kg * self.__conv_length2mm / self.__conv_time2ms**2) * self.__conv_length2mm
        LOG.debug("Conversion [%s] -> [Nm] = %s", self.__unit_moment, conv)
        return conv

    def velocity2ms(self) -> float:
        """Conversion factor for velocity from input unit system to unit  [m/s] (equals [mm/ms])

        Returns:
            float: conversion factor
        """
        conv = self.__conv_length2mm / self.__conv_time2ms
        LOG.debug("Conversion [%s] -> [m/s]/[mm/ms] = %s", self.__unit_velocity, conv)
        return conv

    def acceleration2g(self) -> float:
        """Conversion factor for acceleration from input unit system to unit [g]

        Returns:
            float: conversion factor
        """
        conv = (self.__conv_length2mm * 1000) / (self.__conv_time2ms**2 * scipy.constants.g)
        LOG.debug("Conversion [%s] -> [g] = %s", self.__unit_acceleration, conv)
        return conv

    def pressure2kpa(self) -> float:
        """Conversion factor for pressure from input unit system to unit [kPa]

        Returns:
            float: conversion factor
        """
        conv = (self.__conv_mass2kg * 1000**2) / (self.__conv_time2ms**2 * self.__conv_length2mm)
        LOG.debug("Conversion [%s] -> [kPa] = %s", self.__unit_pressure, conv)
        return conv

    def volume2l(self) -> float:
        """Conversion factor for volume from input unit system to unit [l]

        Returns:
            float: conversion factor
        """
        conv = self.__conv_length2mm**3 / 1000000
        LOG.debug("Conversion [%s] -> [l] = %s", self.__unit_length, conv)
        return conv

    def chest_deflection(
        self,
        dummy: str = "HIII",
        percentile: int = 50,
    ) -> float:
        """Conversion factor for chest deflection from input unit system to unit [mm]
        Use linearization factors if input is [rad]

        Args:
            dummy (str, optional): crash test dummy type. Defaults to "HIII".
            percentile (int, optional): crash test dummy percentile. Defaults to 50.

        Returns:
            float: conversion factor
        """
        if dummy in {"HIII", "H3"}:
            # chest deflection potentiometer linearizing factors from user manual
            # Guha et al. (2011): LSTC Hybrid III 50th Fast Dummy. Positioning & Post‐Processing. Dummy Version: LSTC.H3_50TH_FAST.111130_V2.0, LSTC.
            conv = {5: 96, 50: 145, 95: 158}[percentile]
            LOG.debug("Conversion [rad] -> [mm] = %s", conv)
        else:
            # placeholder assuming THOR measures directly in length unit
            conv = self.__conv_length2mm
            LOG.debug("Conversion [%s] -> [mm] = %s", self.__unit_length, conv)
        return conv
