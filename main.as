// Fish Bowl: a ball on the Customize page, through Cosmetic Kit (a dependency: see info.toml). The ball is a little
// aquarium: a clownfish and a yellow tang swim round it breathing bubbles, over sand with coral, a treasure chest, a
// castle, rocks, a shell and swaying seaweed. The water stays level as the ball rolls.

import bool AddBall(const string &in, const string &in, const string &in, const string &in, const string &in) from "cosmetic-kit";

void Main()
{
    string f = Plugins::Folder();
    AddBall("fish-bowl.aquarium", "Fish Bowl", "", f + "preview.png", f + "models/fish_bowl.txt");
}
