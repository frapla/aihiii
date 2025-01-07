from typing import Literal, Union


class IsoMme:
    def __init__(
        self,
        dummy_type: Literal["H3"],
        dummy_percentile: Literal[5, 50, 95],
        dummy_position: Literal["03"],
    ) -> None:
        """Set channel name string following loosely ISO 22240 MME format
        two additional characters to specify dummy percentile resulting in
        {Position:2s}{Sensor:8s}{Dummy:2s}{Percentile:02d} + {Dimension:2s}{Direction:1s}{FilterClass:1s}

        Args:
            dummy_type (Literal[&quot;H3&quot;]): crash test dummy name
            dummy_percentile (Literal[5, 50, 95]): percentile of dummy
            dummy_position (Literal[&quot;03&quot;]): position in vehicle (e.g. 01 for 1st row left, 03 for 2st row right)
        """
        # dummy
        self.dummy_type = dummy_type
        self.dummy_percentile = dummy_percentile
        self.dummy_position = dummy_position

        # parts
        self._time = "TIME"
        self._no_filt = "X"
        self._acc, self._vel, self._displ = "AC", "VE", "DS"
        self._force, self._moment = "FO", "MO"
        self._directions = ["X", "Y", "Z"]
        self._res_direction = "R"

        # CFC filter
        self.cfc = {
            1000: "A",
            600: "B",
            180: "C",
            60: "D",
            None: "X",
        }  # ISO 6487* / SAE J211:MAR95 1.0

        # sensors (max length 8)
        self._loc_head = "HEAD"
        self._loc_neck = "NECKUP"
        self._loc_chest = "CHST"
        self._loc_pelvis = "PELV"
        self._loc_femur_ri = "FEMRRI"
        self._loc_femur_le = "FEMRLE"
        self._loc_chest_lo_ri = "CHSTRILO"
        self._loc_chest_lo_le = "CHSTLELO"
        self._loc_chest_up_ri = "CHSTRIUP"
        self._loc_chest_up_le = "CHSTLEUP"
        self._loc_shoulder_lo_ri = "SHLDRILO"
        self._loc_shoulder_lo_le = "SHLDLELO"
        self._loc_shoulder_up_ri = "SHLDRIUP"
        self._loc_shoulder_up_le = "SHLDLEUP"
        self._loc_face = "FACE"
        self._loc_knee_ri = "KNEERI"
        self._loc_knee_le = "KNEELE"
        self._loc_belt_b3 = "BELTB3"
        self._loc_cog = "COG"
        self._loc_front = "FRONT"

        # resultants
        self._resultants = {
            self._acc: [self._loc_head, self._loc_chest, self._loc_pelvis],
            self._force: [
                self._loc_chest_lo_ri,
                self._loc_chest_lo_le,
                self._loc_chest_up_ri,
                self._loc_chest_up_le,
                self._loc_shoulder_lo_ri,
                self._loc_shoulder_lo_le,
                self._loc_shoulder_up_ri,
                self._loc_shoulder_up_le,
                self._loc_face,
                self._loc_knee_ri,
                self._loc_knee_le,
            ],
        }

    def channel_name(
        self,
        sensor_loc: str,
        dimension: Literal["AC", "VE", "DS", "FO", "MO"],
        direction: Literal["X", "Y", "Z", "R"],
        cfc: Union[Literal[1000, 600, 180, 60, None], Literal["A", "B", "C", "D", "X"]],
    ) -> str:
        """Generate ISO MME like channel name

        Args:
            sensor_loc (str): sensor location (max. 8 characters)
            dimension (Literal[&quot;AC&quot;, &quot;VE&quot;, &quot;DS&quot;, &quot;FO&quot;, &quot;MO&quot;]): dimension e.g. acceleration (max, 2 characters)
            direction (Literal[&quot;X&quot;, &quot;Y&quot;, &quot;Z&quot;]): spatial direction (1 character)
            cfc (Union[Literal[1000, 600, 180, 60, None], Literal[&quot;A&quot;, &quot;B&quot;, &quot;C&quot;, &quot;D&quot;, &quot;X&quot;]]): CFC filter class

        Returns:
            str: channel name in adapted ISO MME style, e.g. 03CHST0000H350ACXX
        """

        cfc_str = cfc if isinstance(cfc, str) else self.cfc[cfc]

        return f"{self.dummy_position:2s}{sensor_loc:0<8s}{self.dummy_type:0<2s}{self.dummy_percentile:02d}{dimension:0<2s}{direction:1s}{cfc_str:1s}".upper()
