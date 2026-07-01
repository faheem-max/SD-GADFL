


```markdown
# SD-GADFL

SD-GADFL is a decentralized federated learning framework designed for Physical AI Edge Nodes. The system enables Raspberry Pi-based edge devices to communicate, synchronize models, and participate in distributed training without relying on a centralized server.

## Repository Structure

```text
SD-GADFL/
│
├── config/
│   └── peer_config.yaml
│
├── network/
│   ├── http_server.py
│   └── http_client.py
│
├── utils/
│   ├── model_utils.py
│   └── sync_utils.py
│
└── README.md
```
Code Execution Guidelines

Before executing the code, ensure that all required files are placed in the correct directories.

The peer_config.yaml file must be located in the config/ folder. This file contains the peer configuration information required for communication between Physical AI Edge Nodes.

The http_server.py and http_client.py files must be located in the network/ folder. These files handle communication between Raspberry Pi-based edge nodes.

The model_utils.py and sync_utils.py files must be located in the utils/ folder. These files support model handling, utility operations, and synchronization processes.

Deployment on Physical AI Edge Nodes

The complete codebase must be copied to every Physical AI Edge Node participating in the system. Each Raspberry Pi device should contain the same folder structure and the same required files to ensure proper peer-to-peer communication, model synchronization, and decentralized federated learning execution.
