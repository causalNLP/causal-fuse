This repository provides code for the **Guess2Graph** framework, causal discovery methods, and experimental setup developed in the paper:

> [From Guess2Graph: When and How Can Unreliable Experts Safely Boost Causal Discovery in Finite Samples?](https://arxiv.org/abs/2510.14488)  
> **Authors:** Sujai Hiremath, Dominik Janzing, Philipp Faller, Patrick Blöbaum, Elke Kirschbaum, Shiva Prasad Kasiviswanathan, Kyra Gan  
> **Year:** 2025

---

If you find this paper/code useful in your research, we kindly ask you cite the paper as follows:

```bibtex
@article{hiremath2025guess2graph,
  title = {From Guess2Graph: When and How Can Unreliable Experts Safely Boost Causal Discovery in Finite Samples?},
  author = {Hiremath, Sujai and Janzing, Dominik and Faller, Philipp and Blöbaum, Patrick and Kirschbaum, Elke and Kasiviswanathan, Shiva Prasad and Gan, Kyra},
  year = {2025},
  url = {https://arxiv.org/pdf/2510.14488}
}
```

## Repo Organization

Some commentary on the major components of the repo:
- The requirements.txt file can be used to install all dependencies needed to run our code.
- The src folder contains the code for generating synthetic data, downloading real world data, implementing methods, and measuring metrics.
    - For all figures except Figure 2c, here is the correspondence between methods as listed in the paper and function names present in the code: PC and PC-Guess methods use 'run_pc_guess_expert' with different parameters, gPC and gPC-Guess use 'run_sgs_guess_expert' with different parameters, and PC-Stable uses 'run_stable_pc'.
    - For Figure 2c, here is the correspondence between methods as listed in the paper and function names present in the code: PC uses 'run_guess_expert', PC-Stable uses 'run_stable_pc', Claude Opus 4.1 uses the generate_guess_llm.py file in the guess-code folder, gPC uses 'run_sgs_guess_expert', and gPC-Guess + Claude Opus 4.1 uses 'run_sgs_guess_dag'.
- The experiment folder contains the code needed to run all experiments.
- The plotting-code folder contains the code needed to visualize the data collected from running files in the experiment folder.
- The guess-code folder contains the code needed to call LLMs to give DAG guesses using Amazon bedrock.
- The test folder contains tests for verifying accuracy of all major functions.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
