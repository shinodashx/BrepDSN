# BRepDSN

### A Dual-stream Network Design for B-Rep Representation Learning Leveraging the Duality of Face and Boundary Vertices

Haoxuan Song*, Bingchen Yang*, Jun Xiao, Peter Wonka, Haiyong Jiang

*Equal contribution. Haiyong Jiang is the corresponding author and project lead.*

**[Project Page](https://shinodashx.github.io/BrepDSN/)** · **[Code](https://github.com/shinodashx/BrepDSN)**

BRepDSN is a dual-stream network for B-Rep representation learning that leverages complementary features from a Face-adjacency Graph (FG) and a Vertex-adjacency Graph (VG). Bidirectional feature exchange, face-vertex contrastive learning, and auxiliary boundary classification enable boundary-aware and semantically aligned feature encoding.

[![BRepDSN analysis results](docs/assets/teaser.webp)](https://shinodashx.github.io/BrepDSN/)

## Method

- **Dual-stream feature encoding.** The FG branch encodes face adjacency, while the VG branch captures vertex connectivity and boundary structure. A dual-edge interaction module exchanges information between the two streams.
- **Face-vertex contrastive learning.** A face embedding is aligned with the aggregated embedding of its boundary vertices.
- **Boundary classification.** An auxiliary edge classification objective complements face segmentation by identifying edges between faces with different labels.

![BRepDSN network architecture](docs/assets/pipeline.webp)

## Results

BRepDSN is evaluated on modeling operation segmentation, operation step segmentation, B-Rep classification, machining feature segmentation, and assembly mating.

| Dataset | Task | Metric | BRepDSN |
| --- | --- | --- | ---: |
| Fusion360 | Modeling operation segmentation | mIoU | 88.50% |
| CC3D | Modeling operation segmentation | mIoU | 60.35% |
| CC3D-Ops | Operation step segmentation | Step accuracy | 66.23% |
| TMCAD | B-Rep classification | Overall accuracy | 89.68% |
| MFCAD++ | Machining feature segmentation | mIoU | 99.01% |

## Citation

```bibtex
@misc{song_brepdsn,
  title  = {BRepDSN: A Dual-stream Network Design for B-Rep
            Representation Learning Leveraging the Duality
            of Face and Boundary Vertices},
  author = {Song, Haoxuan and Yang, Bingchen and Xiao, Jun
            and Wonka, Peter and Jiang, Haiyong},
  url    = {https://github.com/shinodashx/BrepDSN}
}
```
