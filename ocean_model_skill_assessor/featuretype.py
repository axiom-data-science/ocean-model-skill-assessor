"""All configuration related to NCEI feature types"""


ftconfig = {}
ftconfig["timeSeries"] = {
    "make_time_series": False,
}
ftconfig["profile"] = {
    "make_time_series": False,
}
ftconfig["trajectoryProfile"] = {
    "make_time_series": True,
}
ftconfig["timeSeriesProfile"] = {
    "make_time_series": True,
}
ftconfig["grid"] = {
    "make_time_series": False,
}
