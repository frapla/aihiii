class Names:
    def __init__(self):
        self.perc: str = "Percentile"
        self.env: str = "Environment"
        self.env_coord: str = "Distance"
        self.score: str = "Median Score"
        self.start: str = "Start"
        self.end: str = "End"
        self.feature: str = "FEATURE"
        self.target: str = "TARGET"

        self.chest_ch: str = "Chest a3ms [g]"
        self.head_ch: str = "Head Pos. [mm]"
        self.channel_renamer = {'a3ms_x_acceleration@15-spine_0_mass': self.chest_ch,
                                'min_x_coordinate_local@12-neck': self.head_ch}

        self.chest: str = "Chest"
        self.head: str = "Head"
        self.body: str = "Both"
        self.mid: str = "Mid"
        self.lower: str = "Low"
        self.upper: str = "Up"
        self.range: str = "Range"
        self.double: str = "Double"

        self.bmi: str = "BMI"
        self.stature: str = "Stature"
        self.anthro_para = [self.bmi, self.stature]

        self.sll: str = "Shoulder_Belt_Load_Limiter"
        self.speed: str = "Impact_Speed"
        self.backrest: str = "Backrest_Angle"
        self.seat_cushion: str = "Seat_Cushion_Angle"
        self.envi_para = [self.sll, self.speed, self.backrest, self.seat_cushion]

        self.calibrator: str = "Calibrator"
        self.f_sensor: str = "Feature Sensor"
        self.t_anthro: str = "Target Anthropometric"
        self.t_sensor: str = "Target Sensor"
        self.samples: str = "Samples"

        self.train: str = "Training"
        self.test: str = "Testing"
        self.val: str = "Validation"
        self.pred_type: str = 'Prediction Type'
        self.labels_2, self.labels_3, self.labels_4, self.labels_5 = [f"{i} Labels" for i in range(2, 6)]
        self.regr: str = "Regression"
        self.true: str = "True"
        self.pred: str = "Pred"
        self.median_sc: str = "Median Score"
        self.tt_dist: str = "Train Test Range"
        self.s_name: str = "Score"

        self.hidden_layer: str = "HL"
        self.neurons: str = "NR"
        self.hl_size: str = "Hidden_Layer_Size"
