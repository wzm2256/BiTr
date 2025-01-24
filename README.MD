
-----

#### I am looking for a machine learning engineer/ machine learning researcher position in EU. If you know any opening position, please do not hesitate to contact me! My linkedin page: [Ziming Wang](https://www.linkedin.com/in/ziming-wang-50856916a/)
-----


# SE(3)-bi-equivariant Transformers for Point Cloud Assembly
This is the official implement of the NeurIPS24 paper "SE(3)-bi-equivariant Transformers for Point Cloud Assembly".

## BITR
BITR (SE(3)-bi-equivariant Transformers) is used to solve point cloud assembly problems, where the goal is to align two pieces of point clouds (may not be overlapped).


|Overlapped point clouds| Non-overlapped point clouds| 
|--------------|--------------|
<img src="readme_fig\fig1_1_full.png" width="256"/>  | <img src="readme_fig\fig1_3_full.jpg" width="256"/> |

BITR has 3 nice properties:
1. BITR is **SE(3)-bi-equivariant**: its performance is not influence by the initial input position. This implies that it does not need to be iterative like ICP.
2. BITR is **swap-equivariant**: its result is consistent when the inputs are swapped.
3. BITR is **swap-equivariant**: its result is consistent when the inputs are scaled.

In addition, it does not assume that the inputs are overlapped.


### Requirement
1. dgl (for graph processing)
2. escnn (for equivariant network definition)
3. torch
4. einops
5. torch_cluster, torch_scatter
6. open3d
   

### Usage
Please see `script.txt` for evaluation on 7-scenes, ASL, ShapeNet (airplane) and BB (wine bottle) datasets.



### Checkpoints and Datasets
1. Checkpoints can be downloaded from https://drive.google.com/file/d/13PWIcqmWm42w6tHlzPTMyoCY5nl0WJbZ/view?usp=sharing . They should be placed at the `./saved_model` folder.
2. The processed dataset can be downloaded from https://drive.google.com/file/d/13PWIcqmWm42w6tHlzPTMyoCY5nl0WJbZ/view?usp=sharing. They should be placed at the `./Data` folder.



### Reference

If you find the code useful, please cite the following paper.

    @inproceedings{wang2024se,
    title={SE (3)-bi-equivariant Transformers for Point Cloud Assembly},
    author={Wang, Ziming and J{\"o}rnsten, Rebecka},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
    year={2024},
    }

If you have any question, comment or thought, you are welcome to contact me at wzm2256@gmail.com
