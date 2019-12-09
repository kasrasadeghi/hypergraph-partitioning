#include <stdio.h>    // puts, printf
#include <string>     // std::string
#include <vector>     // vector

#include <fcntl.h>    // open
#include <unistd.h>   // lseek
#include <sys/mman.h> // mmap
#include <assert.h>   // assert

#include <algorithm>  // sort, unique
#include <iostream>   // spectra output

// Spectra
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>


///  OUTLINE
// make hypergraph from reading file
// turn hypergraph into clique graph laplacian (dictionary of keys)
// get 2 smallest eigenvalues/eigenvector pairs of laplacian
// find median in second smallest eigenvector
// partition nodes based on median


//constexpr const char* FILENAME = "xyc.hgr";
constexpr const char* FILENAME = "ISPD98_ibm01.hgr";


using node = int;
using hyperedge = std::vector<node>;
using hypergraph = std::vector<hyperedge>;

// "5" -> 5
// "512" -> 512
int
string_to_int(const std::string& s) {
  int result = 0;
  for (char c : s) {
    result += (c - '0');
    result *= 10;
  }
  result /= 10;
  return result;
}

void
test_string_to_int() {
  printf("%d\n", string_to_int("512"));  
}


hyperedge
parse_line(const std::string& line) {
  
  std::string acc = "";
  hyperedge he;
  int num_count = 0;
  for (char c : line) {
    if (c == ' ') {
      int num = string_to_int(acc);
      he.push_back(num);
      ++ num_count;
      acc = "";
    } else {
      acc += c;
    }
  }

  if (acc != "") {
    int num = string_to_int(acc);
    he.push_back(num);
    ++ num_count;
  }
  

  return std::move(he);
}

void
print_hyperedge(const hyperedge& he) {
  printf("[");
  for (int i = 0; i < he.size(); ++i) {
    if (i != 0) {
      printf(" ");
    }
    printf("%d", he[i]);
  }
  printf("]\n");
}


void
main_hg_check_node_contiguity() {
  int fd = open(FILENAME, O_RDONLY);
  int file_size = lseek(fd, 0, SEEK_END);
  char* data = (char*)mmap(0, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  char* data_end = data + file_size;
  
  char* iter = data;
  std::string acc = "";

  while (*iter != '\n') {
    acc += *iter;
    ++iter;
  }
  ++iter;
  hyperedge he = parse_line(acc);
  assert(he.size() == 2);
  
  size_t node_count = he[0];
  size_t edge_count = he[1];
  printf("node count: %d, edge count: %d\n", node_count, edge_count);

  acc = "";
  
  hypergraph hg;

  while (iter < data_end) {
    char c = *iter;
    if (c == '\n') {
      hg.push_back(parse_line(acc));

      acc = "";
    } else {
      acc += c;
    }
    ++iter;
  }

  print_hyperedge(hg[0]);
  print_hyperedge(hg[1]);
  print_hyperedge(hg[2]);
  print_hyperedge(hg[3]);
  print_hyperedge(hg[hg.size() - 1]);

  assert(hg.size() == edge_count);

  std::vector<node> nodes;
  for (const hyperedge& he : hg) {
    for (node n : he) {
      nodes.push_back(n);
    }
  }

  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

  for (int i = 1; i < nodes.size(); ++i) {
    assert(i+1 == nodes[i]);
  }

  printf("%lu\n", nodes.size());

  assert(nodes.size() == node_count);

  auto adjlist = std::vector<std::vector<node>> {node_count + 1, std::vector<node>()};
  for (hyperedge& he : hg) {
    for (node u : he) {
      for (node v : he) {
        auto contains =
          [](std::vector<node> v, node e)
          { return v.end() != std::find(v.begin(), v.end(), e); };

        if (not contains(adjlist[u], v)) {
          adjlist[u].push_back(v);
        }
      }
    }
  }

  print_hyperedge(adjlist[0]);
  print_hyperedge(adjlist[1]);
  print_hyperedge(adjlist[2]);
  puts("...");
  print_hyperedge(adjlist[adjlist.size() - 2]);
  print_hyperedge(adjlist[adjlist.size() - 1]);
}

void
main_hg() {
  int fd = open(FILENAME, O_RDONLY);
  int file_size = lseek(fd, 0, SEEK_END);
  char* data = (char*)mmap(0, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  char* data_end = data + file_size;
  
  char* iter = data;
  std::string acc = "";

  while (*iter != '\n') {
    acc += *iter;
    ++iter;
  }
  ++iter;
  hyperedge he = parse_line(acc);
  assert(he.size() == 2);
  
  size_t node_count = he[0];
  size_t edge_count = he[1];
  printf("node count: %d, edge count: %d\n", node_count, edge_count);

  acc = "";
  
  hypergraph hg;


  while (iter < data_end) {
    char c = *iter;
    if (c == '\n') {
      hg.push_back(parse_line(acc));

      acc = "";
    } else {
      acc += c;
    }
    ++iter;
  }

  
  auto adjlist = std::vector<std::vector<node>> {node_count + 1, std::vector<node>()};
  for (hyperedge& he : hg) {
    for (node u : he) {
      for (node v : he) {
        auto contains =
          [](std::vector<node> v, node e)
          { return v.end() != std::find(v.begin(), v.end(), e); };

        if (u == v) {
          continue;
        }
        
        if (not contains(adjlist[u], v)) {
          adjlist[u].push_back(v);
        }
      }
    }
  }

  Eigen::SparseMatrix<double> M(node_count + 1, node_count + 1);
  M.reserve(Eigen::VectorXi::Constant(node_count + 1, 700));
  for (int i = 1; i < adjlist.size(); ++i) {
    M.insert(i, i) = adjlist[i].size();
    for (node v : adjlist[i]) {
      M.insert(i, v) = -1;
    }    
  }

  puts("done making sparse matrix laplacian");

  constexpr int eig_count = 2;
  constexpr int conv_rate = eig_count * 10;

  Spectra::SparseGenMatProd<double> op(M);
  Spectra::SymEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseGenMatProd<double>> eigs(&op, eig_count, conv_rate);
  eigs.init();
  int nconv = eigs.compute();

  switch(eigs.info()) {
  case Spectra::NOT_CONVERGING:  puts("ERROR: not converging");  break;
  case Spectra::NOT_COMPUTED:    puts("ERROR: not computed");    break;
  case Spectra::NUMERICAL_ISSUE: puts("ERROR: numerical issue"); break;
  }

  std::vector<double> ev2;

  if (eigs.info() == Spectra::SUCCESSFUL) {
    Eigen::MatrixXd em = eigs.eigenvectors(eig_count);
    // std::cout << em << std::endl;

    for (int i = 0; i < em.rows(); ++i) {
      std::cout << em(i, 1);
    }
    
    std::cout << "eigenvalues: \n" << eigs.eigenvalues().transpose() << std::endl;
  }
}

int
main() {
  //main_hg_check_node_contiguity();
  main_hg();
}
