# Developer documentation

## Updating docs

The demo notebooks are compiled by ReadTheDocs with Myst-NB and jupytext. These packages allow for a 1-1 mapping between a Jupyter notebook and a markdown file that can be interpreted for compilation. The markdown version of each demo is what is git-tracked because changes can be tracked better in that format. Note that currently a couple of the notebooks are in fact being stored as Jupyter notebooks so that they can be run locally.

The steps for updating demo pages:
1. If you don't already have a notebook version of the demo you want to update, convert from markdown to notebook with `jupytext [filename].md --to ipynb`.
2. Update notebook.
3. Convert to markdown with `jupytext [filename].ipynb --to myst`.
4. Git commit and push the markdown file.


## Vocab demo page

The docs page {doc}`add_vocab` is set up differently from the other pages because it contains a widget. Most of the page is an ipython notebook that has been converted to markdown with `Myst-NB`, but the final cell in the page is a raw .html file. One can save the widget state in Jupyter lab under "Settings > Save widget state automatically". It is supposed to be possible to then present the basic widget state with Sphinx but I was not able to get it to work with this setup. However, I was able to export from the "vocab_widget.ipynb" notebook to html and have that properly retain the widget state. As a workaround, then, I embedded the html version of "vocab_widget" in add_vocab.md. Hopefully it will not need to be changed often. That is also why a .ipynb file is saved for the "vocab_widget" demo.


## Roadmap

Next steps:

* Extend to be able to compare model output with other dataset types:
  * time series at other depths than surface
  * gliders
  * 2D surface fields and other depth slices
