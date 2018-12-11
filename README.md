# Label Space Dimension Reduction
## Dataset
[http://mulan.sourceforge.net/datasets-mlc.html][1]

### yeast
|name | domain | instances |nominal	|numeric|labels|cardinality	|density|distinct|
| ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |
| yeast| biology | 2417	 |0|103	|14|4.237|0.303	|198|

## Evaluation
|evaluation criterion |plst | cplst|ECC|PCC(效果很差)|
|---|---|---|---|---|
|hamming loss|
|ranking loss|
|one error|

## Requrements
- Python 3.6
- numpy 1.13.3
- scikit-learn 0.19.1

## Parameter
- plst and cplst regularization parameter:0.1

## Reference
[F. Tai and H.-T. Lin. Multi-Label classification with principal label space transformation. In Neural Computation, 2012.][2]

[Y.-N. Chen and H.-T. Lin, “Feature-aware label space dimension reduction for multi-label classification,” in NIPS, 2012, pp. 1529–1537][3]

  [1]: http://mulan.sourceforge.net/datasets-mlc.html
  [2]: https://www.mitpressjournals.org/doi/abs/10.1162/NECO_a_00320
  [3]: http://papers.nips.cc/paper/4561-feature-aware-label-space-dimension-reduction-for-multi-label-classification




