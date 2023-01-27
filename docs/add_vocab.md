---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3.10.8 ('omsa')
  language: python
  name: python3
---

# How to make and work with vocabularies

This page demonstrates the workflow of making a new vocabulary, saving it to the user application cache, and reading it back in to use it. The vocabulary created is the exact same as the "general" vocabulary that is saved with the OMSA package, though here it is given another name to demonstrate that you could be making any new vocabulary you want.

Here is the list of variables of interest (with "nickname"), aimed at a physical oceanographer, which are built into the vocabulary:

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
* sea ice velocity u "sea_ice_u"
* sea ice velocity v "sea_ice_v"
* sea ice area fraction "sea_ice_area_fraction"

```{code-cell} ipython3
import cf_pandas as cfp
import ocean_model_skill_assessor as omsa
import pandas as pd
```

## Vocabulary workflow

### Make vocabulary

Here we show making the "general" vocabulary that is saved into the repository. This is a more general vocabulary to identify variables from sources that don't use exact CF standard_names.

```{code-cell} ipython3
nickname = "temp"
vocab = cfp.Vocab()

# define a regular expression to represent your variable
reg = cfp.Reg(include_or=["temp","sst"], exclude=["air","qc","status","atmospheric","bottom"])

# Make an entry to add to your vocabulary
vocab.make_entry(nickname, reg.pattern(), attr="name")

vocab.make_entry("salt", cfp.Reg(include_or=["sal","sss"], exclude=["soil","qc","status","bottom"]).pattern(), attr="name")
vocab.make_entry("ssh", cfp.Reg(include_or=["sea_surface_height","surface_elevation"], exclude=["qc","status"]).pattern(), attr="name")

reg = cfp.Reg(include=["east", "vel"])
vocab.make_entry("u", "u$", attr="name")
vocab.make_entry("u", reg.pattern(), attr="name")

reg = cfp.Reg(include=["north", "vel"])
vocab.make_entry("v", "v$", attr="name")
vocab.make_entry("v", reg.pattern(), attr="name")

reg = cfp.Reg(include=["up", "vel"])
vocab.make_entry("w", "w$", attr="name")
vocab.make_entry("w", reg.pattern(), attr="name")

vocab.make_entry("water_dir", cfp.Reg(include=["dir","water"], exclude=["qc","status","air","wind"]).pattern(), attr="name")

vocab.make_entry("water_speed", cfp.Reg(include=["speed","water"], exclude=["qc","status","air","wind"]).pattern(), attr="name")

vocab.make_entry("wind_dir", cfp.Reg(include=["dir","wind"], exclude=["qc","status","water"]).pattern(), attr="name")

vocab.make_entry("wind_speed", cfp.Reg(include=["speed","wind"], exclude=["qc","status","water"]).pattern(), attr="name")

reg1 = cfp.Reg(include=["sea","ice","u"], exclude=["qc","status"])
reg2 = cfp.Reg(include=["sea","ice","x","vel"], exclude=["qc","status"])
reg3 = cfp.Reg(include=["sea","ice","east","vel"], exclude=["qc","status"])
vocab.make_entry("sea_ice_u", reg1.pattern(), attr="name")
vocab.make_entry("sea_ice_u", reg2.pattern(), attr="name")
vocab.make_entry("sea_ice_u", reg3.pattern(), attr="name")

reg1 = cfp.Reg(include=["sea","ice","v"], exclude=["qc","status"])
reg2 = cfp.Reg(include=["sea","ice","y","vel"], exclude=["qc","status"])
reg3 = cfp.Reg(include=["sea","ice","north","vel"], exclude=["qc","status"])
vocab.make_entry("sea_ice_v", reg1.pattern(), attr="name")
vocab.make_entry("sea_ice_v", reg2.pattern(), attr="name")
vocab.make_entry("sea_ice_v", reg3.pattern(), attr="name")

vocab.make_entry("sea_ice_area_fraction", cfp.Reg(include=["sea","ice","area","fraction"], exclude=["qc","status"]).pattern(), attr="name")

vocab
```

### Save it

This exact vocabulary was previously saved as "general" and is available under that name, but this page demonstrates saving a new vocabulary and so we use the name "general2" to differentiate.

```{code-cell} ipython3
vocab.save(omsa.VOCAB_PATH("general2"))
```

```{code-cell} ipython3
omsa.VOCAB_PATH("general2")
```

### Use it later

Read the saved vocabulary back in to use it:

```{code-cell} ipython3
vocab = cfp.Vocab(omsa.VOCAB_PATH("general2"))

df = pd.DataFrame(columns=["sst", "time", "lon", "lat"], data={"sst": [1,2,3]})
with cfp.set_options(custom_criteria=vocab.vocab):
    print(df.cf["temp"])
```

## Combine vocabularies

A user can add together vocabularies. For example, here we combine the built-in "standard_names" and "general" vocabularies.

```{code-cell} ipython3
v1 = cfp.Vocab(omsa.VOCAB_PATH("standard_names"))
v2 = cfp.Vocab(omsa.VOCAB_PATH("general"))

v = v1 + v2
v
```

## Using the `cf-pandas` widget

+++

.. raw:: html
   <iframe src="vocab_widget.html" height="500px" width="100%"></iframe>
