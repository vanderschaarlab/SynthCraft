import torch

# make sure uint{16,32,64} exist (point at their signed cousins)
for unsigned, signed in (
    ("uint16", "int16"),
    ("uint32", "int32"),
    ("uint64", "int64"),
):
    if not hasattr(torch, unsigned):
        setattr(torch, unsigned, getattr(torch, signed))

import streamlit as st

import climb.ui.st_common as st_common

st_common.menu()

# NOTE: This page should not end up being displayed to the user (we auto-navigate to `main.py` page), but just in case:
st.markdown("⇦ Please select a page from the sidebar.")
