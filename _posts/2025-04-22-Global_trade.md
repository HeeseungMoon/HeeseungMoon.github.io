---
layout: post
title:  "Global Trade Networks through Rice and Wheat: A Data-Driven View of Global Flows"
---

<div style="text-align: center;">
    This project is from my class. Feel free to look around.
</div>

<p>
  📄 <a href="https://github.com/HeeseungMoon/HeeseungMoon.github.io/raw/master/assets/(2021) Deng et al.pdf" target="_blank">Download the PDF</a>
</p>

<div style="background-color:#f8f8f8; padding:15px; border-radius:8px;">
<pre><code>
import pandas as pd

df = pd.read_csv('data.csv')
df.head()
</code></pre>
</div>

```python
import networkx as nx
import pandas as pd

df = pd.read_csv("data.csv")
G = nx.from_pandas_edgelist(df, source="source", target="target")
'''
