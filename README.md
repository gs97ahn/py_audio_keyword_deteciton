# Speech Keyword Detection
<div align="center">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="Tensorflow">
</div>

**Speech keyword detection is a deep learning model that recognizes a keyword when spoken.**

## Dependencies
To run this program, the following dependencies need to be installed.
```commandline
pip install -r requirements.txt
```

## Models
- DNN
- CNN
- ResNet3

## Environment
The experiment was designed specifically to run on a small embedded system such as NVIDIA Jetson Nano 2GB.

## Dataset
In this experiment, the dataset was created manually. It is created with three different mic location and three
different reverberation time, providing nine different combination. Also, it is combined with six different genre of TV 
programs.
- Training dataset = 24,006
- Testing dataset = 31.968

## Result
CNN with no-normalization and log mels feature extracted was the best model in this experiment.
<table>
<thead>
  <tr>
    <th>Normalization</th>
    <th>Feature</th>
    <th >Model</th>
    <th>Validation Accuracy</th>
    <th>EER</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="6">No-Normalization</td>
    <td rowspan="3">Log Mels</td>
    <td>CNN</td>
    <td>95.75%</td>
    <td>4.04%</td>
  </tr>
  <tr>
    <td>FC DNN</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ResNet</td>
    <td>88.59%</td>
    <td>13.60%</td>
  </tr>
  <tr>
    <td rowspan="3">MFCC</td>
    <td>CNN</td>
    <td>77.43%</td>
    <td>15.32%</td>
  </tr>
  <tr>
    <td>FC DNN</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ResNet</td>
    <td>92.08%</td>
    <td>7.91%</td>
  </tr>
  <tr>
    <td rowspan="6">Max-Normalization</td>
    <td rowspan="3">Log Mels</td>
    <td>CNN</td>
    <td>89.15%</td>
    <td>13.37%</td>
  </tr>
  <tr>
    <td>FC DNN</td>
    <td>91.58%</td>
    <td>8.30%</td>
  </tr>
  <tr>
    <td>ResNet</td>
    <td>81.06%</td>
    <td>18.58%</td>
  </tr>
  <tr>
    <td rowspan="3">MFCC</td>
    <td>CNN</td>
    <td>81.13%</td>
    <td>15.78%</td>
  </tr>
  <tr>
    <td>FC DNN</td>
    <td>90.33%</td>
    <td>17.18%</td>
  </tr>
  <tr>
    <td>ResNet</td>
    <td>87.24%</td>
    <td>12.84%</td>
  </tr>
</tbody>
</table>