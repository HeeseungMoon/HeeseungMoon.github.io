---
layout: post
title:  "Global Trade Networks through Rice and Wheat: A Data-Driven View of Global Flows"
---

<div style="text-align: center;">
    This project is from my class. Feel free to look around.
</div>

<div style="font-size: 24px;">
    Source
</div>

<p>
  📄 <a href="https://github.com/HeeseungMoon/HeeseungMoon.github.io/raw/master/assets/(2021) Deng et al.pdf" target="_blank">(2021) Deng et al.pdf.</a>
</p>

<div style="font-size: 24px;">
  <b>1. Summary of the reading</b>
</div>

<div>
  <b> •Research Question & Research Gap</b>
</div>

<div>
Research question: How does virtual water flow between countries, and how can we use social network analysis (SNA) to understand the global structure of this flow?
</div>

<div>
Research Gap: Most past research focused only on agricultural products and didn’t explore the whole-industry level or apply SNA to network characteristics.
</div>



```python
import networkx as nx
import pandas as pd

df = pd.read_csv("data.csv")
G = nx.from_pandas_edgelist(df, source="source", target="target")
```
