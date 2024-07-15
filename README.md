
# CTGAN: Conditional Tabular GAN

CTGAN is a PyTorch implementation of Conditional Tabular Generative Adversarial Networks for generating synthetic data.

## Features

- Generates synthetic tabular data
- Handles both discrete and continuous columns
- Conditional generation based on specified attributes
- Efficient training using PyTorch

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from ctgan import CTGAN
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
discrete_columns = ['column1', 'column2']

# Initialize and train the model
ctgan = CTGAN(epochs=300, batch_size=500)
ctgan.fit(data, discrete_columns)

# Generate synthetic data
synthetic_data = ctgan.sample(100)
```

## Documentation

For detailed documentation, please visit [our documentation page](https://your-ctgan-docs-url.com).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
