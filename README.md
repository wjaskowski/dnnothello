This repo contains some code used to run experiments for 
[Learning to Play Othello with Deep Neural Networks](https://arxiv.org/abs/1711.06583) by Paweł Liskowski, Wojciech Jaśkowski, and Krzysztof Krawiec, 2018.

**Disclaimer**: None of the authors remembers how the code should be executed or whether it still works at all. 
Help your self but you are on your own; expect no help nor mercy.

## Boards data

> othello_data.py generate

To use it:

othello_data.get_learning_data()

## Benchmark vs. Edax

### Prepare:
> wget http://abulmo.perso.neuf.fr/edax/4.3/edax.4-3-2.zip
> unzip edax.4-3-2.zip
> cd edax/4.3.2/src: 
> make COMP=clang OS=osx

> wget http://abulmo.perso.neuf.fr/edax/book.21.2011-1-11.7z
> extract to edax/bin/data and save at book_good.dat

### Benchmark:
* Change EDAX_BIN_PATH in games/benchmark.py
* Change & execute main

### Resources

* WTHOR database: http://www.ffothello.org/informatique/la-base-wthor/
* Format description (in French): http://www.ffothello.org/wthor/Format_WThor.pdf
* Blog description in English: http://ledpup.blogspot.com/2012/03/computer-reversi-part-15-thor-database.html
