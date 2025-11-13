# XuanHeng Modifications

commit1:

Fixed the warnings in isaacsim app:

in blind_env_cfg and velocity_env_cfg, change attach_yaw_only to ray_alignment="yaw", it says attach_yaw_only will be deprated
add usd folder to data/Robots/Unitree/Go2/go2_descrption and modified unitree.py, so that meshes are correctly loaded.

Script modis:

in train.py, comment out blockes of code which seems for debug. Add parser train_privileged_agent and deprate TEACHER_POLICY_AVAILABLE "try import" logic.
*btw, I think that txl agent should not be trained aby a ppo runner. I'll figure it out later.*

IDE：
in pyproject.toml, add env paths under tool.pyright
其余的修改是我嫌IDE报的那些格式警告太丑了

commit2:
直接从郭靖43dfe32分支复制了velocity文件夹到本地。我看郭靖加了一个terrains文件夹，别的还加了什么我不知道，害怕出bug就全拉过来了。
尝试改了一下commands
