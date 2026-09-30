# Fish Bowl

A plugin for the [Ballest plugin manager](https://github.com/AnythingGoes-ballest/ballest-plugin-manager): a ball that
is an aquarium, added to the **balls** tab of the **Customize** page through
[Cosmetic Kit](https://github.com/AnythingGoes-ballest/ballest-cosmetic-kit).

A clownfish and a yellow tang swim laps round the tank, one each way, breathing little bubbles that rise and pop. The
tank has a sand floor with four corals, a treasure chest puffing bubbles, a castle, rocks, a shell and four strands of
swaying seaweed. The water stays level and everything stays upright while the ball rolls.

## Install

In the game: footer **plugins** > **browse** > Fish Bowl > **install** (Cosmetic Kit comes with it).
Needs the plugin manager host 0.17.1 or newer.

## Credits

The models are CC0 (public domain), from [Poly Pizza](https://poly.pizza):

- [Quaternius](https://quaternius.com): the clownfish, the yellow tang, the ocean chest, the rocks and the tower.
- MiniPoly: the Coral Reef Set (four of its corals).
- [Kenney](https://kenney.nl): the shell (Mussel Open).

The sand, the bubbles, the seaweed and the water are made here by `make_bowl.py`.

## Files

- `main.as`: adds the ball.
- `models/fish_bowl.txt`: the model, in the format of the plugin manager's
  [custom cosmetics guide](https://anythinggoes-ballest.github.io/ballest-plugin-manager/guides/cosmetics/).
- `models/*.glb`, `preview.png`, `icon.png`: made by `make_bowl.py` (Python with NumPy and Pillow) from `sources/`.
  `preview.py` draws the tile picture.

## License

MIT for this plugin's code and files; the models keep their CC0 licence.
