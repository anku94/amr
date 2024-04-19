class Rank {
 public:
  int id;
  double load;

  Rank(int id, double load) : id(id), load(load) {}
};

class LeastLoadedRankPQComparator {
 public:
  bool operator()(Rank& a, Rank& b) { return a.load > b.load; }
};
