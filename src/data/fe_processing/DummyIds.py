from IsoMme import IsoMme


class DummyIds:
    def __init__(self, mme: IsoMme) -> None:
        """IDs for FE model of HIII

        Args:
            mme (IsoMme): object with ISO MME like naming convention
        """
        if mme.dummy_type == "H3" and mme.dummy_position == "03":
            self._nodout_ids = {
                "68000001": mme._loc_head,
                "68001787": mme._loc_chest,
                "68000736": mme._loc_chest,  # original fe-model
                "68003304": mme._loc_pelvis,
                "68002091": mme._loc_pelvis,  # original fe-model
            }
            self._jntforc_f_ids = {
                "68000039": mme._loc_neck,
                "68000025": mme._loc_femur_ri,
                "68000024": mme._loc_femur_le,
            }
            self._jntforc_m_ids = {
                "68000044": mme._loc_neck,
                "68000026": mme._loc_neck,  # original fe-model
            }
            self._deforc_ids = {
                "68000010": mme._loc_chest,
                "68004276": mme._loc_chest,  # original fe-model
            }
            self._rcforc_ids = {
                "68000701s": mme._loc_chest_lo_ri,
                "68000702s": mme._loc_chest_lo_le,
                "68000703s": mme._loc_chest_up_ri,
                "68000704s": mme._loc_chest_up_le,
                "68000705s": mme._loc_shoulder_lo_ri,
                "68000706s": mme._loc_shoulder_lo_le,
                "68000707s": mme._loc_shoulder_up_ri,
                "68000708s": mme._loc_shoulder_up_le,
                "68000709s": mme._loc_face,
                "68000710s": mme._loc_knee_ri,
                "68000711s": mme._loc_knee_le,
            }
