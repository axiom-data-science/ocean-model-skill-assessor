"""All configuration related to NCEI feature types"""


ftconfig = {}
ftconfig["timeSeries"] = {
    "locstreamT": False,
    "locstreamZ": False,
}
ftconfig["profile"] = {
    "locstreamT": False,
    "locstreamZ": False,
}
ftconfig["timeSeriesProfile"] = {
    "locstreamT": False,
    "locstreamZ": False,
}
ftconfig["trajectory"] = {
    "locstreamT": True,
    "locstreamZ": False,
}
ftconfig["trajectoryProfile"] = {
    "locstreamT": True,
    "locstreamZ": True,
}
ftconfig["grid"] = {
    "locstreamT": False,
    "locstreamZ": False,
}
