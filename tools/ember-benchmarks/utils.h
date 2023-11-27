//
// Created by Ankush J on 11/27/23.
//

#pragma once

int get_logical_rank(const char* map_file, int phy_rank) {
  if (map_file == NULL) {
    printf("No map file provided. Returning physical rank.\n");
    return phy_rank;
  }

  FILE *f = fopen(map_file, "r");
  int a = -1, b = -1;
  while (fscanf(f, "%d,%d\n", &a, &b) != EOF) {
    if (a == phy_rank) {
      break;
    }
  }
  fclose(f);
  return b;
}

int get_physical_rank(const char* map_file, int log_rank) {
  if (map_file == NULL) {
    printf("No map file provided. Returning logical rank.\n");
    return log_rank;
  }

  FILE *f = fopen(map_file, "r");
  int a = -1, b = -1;
  while (fscanf(f, "%d,%d\n", &a, &b) != EOF) {
    if (b == log_rank) {
      break;
    }
  }
  fclose(f);
  return a;
}


