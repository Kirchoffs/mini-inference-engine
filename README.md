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
