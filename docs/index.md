# MoMaGen

**Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation**

![momagen_teaser](https://github.com/user-attachments/assets/278f7c1c-1d73-47c0-942c-e70115875eb4)

## What is MoMaGen?

MoMaGen is a framework for generating several robot demonstrations using a single human-collected demonstration for complex bimanual mobile manipulation tasks. By considering soft constraints—like maintaining visibility of the object during navigation—and hard constraints—such as guaranteeing the object is reachable upon arrival—it produces multi-step demonstrations that are both feasible and useful for policy learning.

## Supported Tasks

- `pick_cup` - Pick and place a cup
- `tidy_table` - Organize objects on a table
- `dishes_away` - Put dishes into storage
- `clean_pan` - Clean and store a pan

## Quick Links

- [Installation Guide](installation.md)
- [Data Generation](./tutorials/generating-data.md)
- [GitHub Repository](https://github.com/ChengshuLi/MoMaGen)

## License

See [LICENSE](https://github.com/ChengshuLi/MoMaGen/blob/main/LICENSE) for details.
