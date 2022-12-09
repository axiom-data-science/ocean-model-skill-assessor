---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Creating vocabularies for known servers and BOEM

This page demonstrates setting up a series of vocabularies matched to known servers and assuming a set of variables of interest. These vocabularies can then be saved and combined to create one for regular use.

Variables of interest:

* water temperature "temp"
* salinity "salt"
* sea surface height "ssh"
* u velocity "u"
* v velocity "v"
* w upward velocity "w"
* direction of water velocity "water_dir"
* magnitude of water velocity "water_speed"
* wind direction "wind_dir"
* wind speed "wind_speed"
* sea ice concentration "sea_ice_con"
* sea ice velocity u "sea_ice_u"
* sea ice velocity v "sea_ice_v"
* sea ice area fraction "sea_ice_area_fraction"

The workflow for each vocabulary is to

1. enter the variable nickname in the first text box
1. enter strings to include and exclude to filter the list of variable names down
1. select however many options that will be used as exact matches to be equivalent to your variable
1. click save
1. repeat with each variable for vocabulary
1. save to disk

```{code-cell} ipython3
import cf_pandas as cfp
import intake
import intake_erddap
```

## ERDDAP servers

+++

### IOOS

```{code-cell} ipython3
server = "http://erddap.sensors.ioos.us/erddap"
df = intake_erddap.utils.return_category_options(server=server)
```

```{code-cell} ipython3
df['Category']
```

```{code-cell} ipython3
w = cfp.Selector(options=df['Category'])
```

```{code-cell} ipython3
# w.vocab.save("vocabs/erddap_ioos")
```

### Coastwatch

```{code-cell} ipython3
server = "https://coastwatch.pfeg.noaa.gov/erddap"
df = intake_erddap.utils.return_category_options(server=server)
w2 = cfp.Selector(options=df['Category'])
```

```{code-cell} ipython3
w2.vocab.save("vocabs/erddap_coastwatch")
```

## Axiom resources

```{code-cell} ipython3

```

## CF Standard Names

In the best case scenarios, model and data variables will have a standard_name associated with them, so we will also create a vocabulary that can be used to exactly match those names.

```{code-cell} ipython3
sn = cfp.standard_names()
w3 = cfp.Selector(options=sn)
```

```{code-cell} ipython3
w3.vocab.save("vocabs/standard_names")
```

## General vocabulary

We may need a more general vocabulary to capture variables from other sources that don't use these exact names. Here we make a vocabulary for that purpose.

```{code-cell} ipython3
nickname = "temp"
vocab = cfp.Vocab()

# define a regular expression to represent your variable
reg = cfp.Reg(include="temp", exclude=["air","qc","status","atmospheric"])

# Make an entry to add to your vocabulary
vocab.make_entry(nickname, reg.pattern(), attr="name")

vocab.make_entry("salt", cfp.Reg(include="sal", exclude=["soil","qc","status"]).pattern(), attr="name")
vocab.make_entry("ssh", cfp.Reg(include=["sea_surface_height","surface_elevation"], exclude=["qc","status"]).pattern(), attr="name")


vocab.make_entry("T", cfp.Reg(include=["time"]).pattern(), attr="name")
vocab.make_entry("longitude", cfp.Reg(include=["lon"]).pattern(), attr="name")
vocab.make_entry("latitude", cfp.Reg(include=["lat"]).pattern(), attr="name")

# ADD OTHER VARIABLES
vocab.save("vocabs/general")
```

## Combine vocabularies

A user can add together vocabularies. This vocabulary exactly matches all of the selections we made for the vocabularies above.

For example:

```{code-cell} ipython3
v1 = cfp.Vocab("vocabs/erddap_ioos.json")
v2 = cfp.Vocab("vocabs/erddap_coastwatch.json")
v3 = cfp.Vocab("vocabs/standard_names.json")
```

```{code-cell} ipython3
v = v1 + v2 + v3
v
```
