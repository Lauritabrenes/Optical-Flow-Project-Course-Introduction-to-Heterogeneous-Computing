#!/bin/bash
sudo bash -c "source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; cd /home/el5859/deploy && python3 \$1"
