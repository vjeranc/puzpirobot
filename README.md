# puzpirobot
Bilging bot with simple DFS in Python. Draws stuff that needs to be pressed on screen.

You have to have images for the pieces. I did not upload them.

You can check if the system works by using an image input before you use the screenshoting loop.

## algorithm

Board state tracks piece counts in a way that allows O(1) calculation of cleared pieces when
swapping two pieces. Similarily, O(1) update of the data structure is possible if swap happened
but pieces weren't cleared.

Given this datastructure, it's easy to do DFS and undoing the swap moves to continue to different
branches.

One other optimization is excluding the rows/columns from search that are too far to be affected
by the current swap.
