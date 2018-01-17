rm -f $1.o
rm -f $1

gcc -c $1.c
gcc $1.o -o $1
