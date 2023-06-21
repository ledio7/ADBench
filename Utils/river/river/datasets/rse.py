from river import stream

from . import base


class RSE(base.FileDataset):

    def __init__(self):
        super().__init__(
            n_samples=4_663_910,
            n_features=28,
            task=base.BINARY_CLF,
            filename="rse.zip",
        )

    def bool_to_int(value):
        return int(value)

    def __iter__(self):

        self.converters = {
            "Transformer_ID" :float,
            "KG_TransformerName_CIM"     :float,
            "KG_primarySubstationAlias"  :float,
            "KG_TerminalName"            :float,
            "Real_Power_Mean_kW"         :float,
            "Reactive_Power_Mean_kVar"   :float,
            "Voltage_kV"                 :float,
            "Rain_Day_Cumulated_mm"      :float,
            "Wind_Average_Speed_m_s"     :float,
            "Temperature_Average_c"      :float,
            "measures_dow"               :int,
            "measures_doy"               :int,
            "measures_woy"               :int,
            "measures_h"                 :int,
            "measures_min"               :int,
            "measures_month"             :int,
            "measures_y"                 :int,
            "measures_weekend"           :int,
            "measures_holiday"           :int,
            "weather_dow"                :int,
            "weather_doy"                :int,
            "weather_woy"                :int,
            "weather_h"                  :int,
            "weather_min"                :int,
            "weather_month"              :int,
            "weather_y"                  :int,
            "weather_weekend"            :int,
            "weather_holiday"            :int
        }
        self.converters["label"] = int
        return stream.iter_csv(self.path, target="label", converters=self.converters)