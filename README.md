# Spot RL Deployment Code

## Setup
```bash
sudo apt-get update && sudo apt-get upgrade
conda create -n spot python=3.10
conda activate spot
pip install gitman
gitman update
cd external/spot-sdk/prebuilt
pip install bosdyn_api-4.1.0-py3-none-any.whl
pip install bosdyn_core-4.1.0-py3-none-any.whl
pip install bosdyn_client-4.1.0-py3-none-any.whl
pip install pygame
pip install pyPS4Controller
pip install spatialmath-python
pip install onnxruntime
```

## Convert env.yaml to env.json
```bash
cd python/utils/
python env_convert.py 
```

## Connect PS4 Controller
```bash
bluetoothctl
scan on  // wait for devices populate ~5s
scan off
devices
```
```bash
trust {MAC} 
pair {MAC} 
connect {MAC} 
exit
```

# Run RL policy on Spot
```bash
python ./python/spot_rl_demo.py <spot_ip> ./models --gamepad-config ./python/gamepad_config.json
```
