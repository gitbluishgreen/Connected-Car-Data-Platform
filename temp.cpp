#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

int fd[2];
int matrix[2][2];

int main () {

  pipe (fd);
  if (0 == fork()) {
    printf ("Start child process with pid: %d\n", getpid());
    for (int i = 0; i < 2; i++)
      matrix[i][i] = 1;

    write (fd[1], matrix, 4);
    exit (0);
  }

  printf ("Start parent process with pid: %d\n", getpid());
  read (fd[0], matrix, 4);
  printf ("Received %d\n", matrix[1][1]);

return 0;
}