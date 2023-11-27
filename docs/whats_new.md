# What's New

## v1.2.0 (November 27, 2023)
* Added capability for running HF Radar as quiver plot, over time or single time.

## v1.1.0 (October 13, 2023)
* Continuing to improve functionality of flags to be able to control how model output is extracted
* making code more robust to different use cases

## v1.0.0 (October 5, 2023)
* more modularized code structure with much more testing
* requires datasets to include catalog metadata of NCEI feature type and maptype (for plotting):
  * feature types currently included:
    * timeSeries
    * profile
    * trajectoryProfile
    * timeSeriesProfile
  * To be added: grid
* added option for user to input labels for vocab keys to be used in plots
* configuration for handling featuretypes is in `featuretype.py` and `plot.__init__`.
* Added images-based tests for each featuretype, which can be run to compare against expected images with `pytest --mpl`. There is a developer section in the documentation with instructions.
* Added checks for what DataFrame datasets need to include and what catalog source metadata needs to include
* expanded documentation on requirements and information for datasets and catalogs, including using and understanding NCEI feature types
* Full update of docs

## v0.9.0 (September 15, 2023)
* improved index handling

## v0.8.0 (September 11, 2023)

* `omsa.run` now saves the polygon found for the input model into the project directory.
* bunch of other changes
