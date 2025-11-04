# XuanHeng Modifications

Fixed the warnings in isaacsim app:

in blind_env_cfg and velocity_env_cfg, change attach_yaw_only to ray_alignment="yaw", it says attach_yaw_only will be deprated
add usd folder to data/Robots/Unitree/Go2/go2_descrption and modified unitree.py, so that meshes are correctly loaded.

Script modis:

in train.py, comment out blockes of code which seems for debug. Add parser train_privileged_agent and deprate TEACHER_POLICY_AVAILABLE "try import" logic.
*btw, I think that txl agent should not be trained aby a ppo runner. I'll figure it out later.*

IDE：
in pyproject.toml, add env paths under tool.pyright
其余的修改是我嫌IDE报的那些格式警告太丑了
