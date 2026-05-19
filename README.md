# Notes
## Installation Options
### Conda Install
#### Install
```
>> conda install -c conda-forge armadillo gtest glog sentencepiece -y
>> conda install -c conda-forge pkg-config cmake make -y
```

#### Verify
```
>> ls $CONDA_PREFIX/include/armadillo
>> ls $CONDA_PREFIX/include/gtest
>> ls $CONDA_PREFIX/include/glog
>> ls $CONDA_PREFIX/include | grep sentencepiece
```

```
>> mkdir -p build && cd build
>> cmake ..
>> make
>> ./verify
```

### Docker Install
#### Install
```
>> cd docker
>> docker compose up -d --build
```

#### Verify
```
>> docker exec -it mini-inference-engine-container /bin/bash
(docker) >> mkdir -p build && cd build
(docker) >> cmake ..
(docker) >> make
(docker) >> ./verify
```

### Native Install
#### Install
```
>> chmod +x dependencies_setup.sh
>> ./dependencies_setup.sh
```

#### Verify
```
>> mkdir -p build && cd build
>> cmake ..
>> make
>> ./verify
```

## Armadillo
### Cube
#### Types
```
cube    = Cube<double>
dcube   = Cube<double>
fcube   = Cube<float>
cx_cube = Cube<cx_double>
```

#### Constructors
```
cube(n_rows, n_cols, n_slices)
```

#### Member Functions
```
reshape(n_rows, n_cols, n_slices)
```

## Miscellaneous
### Header Inclusion Order
- Corresponding Header (The file being implemented)
- C System Headers
- C++ Standard Library Headers
- Third-party Library Headers
- Project-specific / Local Headers
