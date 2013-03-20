#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Natural Language Toolkit: NEGRA Corpus Reader
#
# URL: <http://www.experimentallabor.de/>
#
# Copyright 2011 Philipp Nolte
# Copyright 2013 Will Roberts <wildwilhelm@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Read NEGRA corpus files.
"""

from nltk.tree               import Tree, ParentedTree
from nltk.util               import LazyMap
from nltk.util               import LazyConcatenation
from nltk.corpus.reader      import ConllCorpusReader
from nltk.corpus.reader.util import read_regexp_block
from nltk.corpus.reader.api  import CorpusReader

class Atom(object):
    '''
    An object which acts like a bare string, but additionally contains
    properties representing the part of speech, morphology and
    syntactic parent of a token.
    '''
    def __init__(self, word, tag, morph, grid_lineno, parent):
        self.word        = word
        self.tag         = tag
        self.morph       = morph
        self.grid_lineno = grid_lineno
        self._parent     = parent
    def __str__(self):
        return str(self.word)
    def __unicode__(self):
        return unicode(self.word)
    def __repr__(self):
        return repr(self.word)
    def __len__(self):
        return len(self.word)
    def __getitem__(self, key):
        return self.word[key]
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.word == other.word
        return False
    def __ne__(self, other):
        return not self.__eq__(other)
    def parent(self):
        '''Returns the tree node which is parent to this Atom.'''
        return self._parent

class NegraCorpusReader(ConllCorpusReader):
    """A corpus reader for NEGRA corpus files. A NEGRA corpus file consists out
    of annotated sentences separated by #BOS (beginning of sentence) and #EOS
    (end of sentence) markers on their own line. Each sentence consists of
    words and columns containing the lemma, tag, chunk and/or morphological tag.
    Each word and its tag information has its own line. Eg. the form

    %% word     lemma       tag     parent
    #BOS 1
    The         the         DET     500
    house       house       N       500
    is          be          V       501
    red         red         ADJ     501
    .           --          .       502
    #500        --          NP      502
    #501        --          VP      502
    #502        --          S       0
    #EOS 1

    Because of similar corpus structure, this reader is based on and very
    similar to the ConllCorpusReader. Both corpora have their tokens structured
    as grid. NEGRA corpus file provide more token information though.
    """

    #==========================================================================
    # Default column types
    #==========================================================================
    WORDS   = 'words'   # Column for the word
    LEMMA   = 'lemma'   # Column for the lemma
    POS     = 'pos'     # Column for the tag
    MORPH   = 'morph'   # Column for the morphological tag
    EDGE    = 'edge'    # Column for the grammatical function
    PARENT  = 'parent'  # Column for the words parent in the sentence tree
    SECEDGE = 'secedge' # Column for an optional second grammatical function
    COMMENT = 'comment' # Column for an optional comment from the editor

    # List of supported column types
    COLUMN_TYPES = (WORDS, LEMMA, POS, MORPH, EDGE, PARENT, SECEDGE, COMMENT)

    #==========================================================================
    # Constructor
    #==========================================================================

    def __init__(self,
                 root,
                 fileids,
                 column_types=None,
                 top_node='S',
                 beginning_of_sentence=r'#BOS.+$',
                 end_of_sentence=r'#EOS.+$',
                 encoding=None):
        """ Construct a new corpus reader for reading NEGRA corpus files.
        @param root: The root directory of the corpus files.
        @param fileids: A list of or regex specifying the files to read from.
        @param column_types: An optional C{list} of columns in the corpus.
        @param top_node: The top node of parsed sentence trees.
        @param beginning_of_sentence: A regex specifying the start of a sentence
        @param end_of_sentence: A regex specifying the end of a sentence
        @param encoding: The default corpus file encoding.
        """

        # Make sure there are no invalid column type
        if isinstance(column_types, list):
            for column_type in column_types:
                if column_type not in self.COLUMN_TYPES:
                    raise ValueError("Column %r is not supported." % columntype)
        else:
            column_types = self.COLUMN_TYPES

        # Define stuff
        self._top_node = top_node
        self._column_types = column_types
        self._fileids = fileids
        self._bos = beginning_of_sentence
        self._eos = end_of_sentence
        self._colmap = dict((c,i) for (i,c) in enumerate(column_types))

        # Finish constructing by calling the extended class' constructor
        CorpusReader.__init__(self, root, fileids, encoding)

    #==========================================================================
    # Data access methods
    #==========================================================================

    def lemmatised_words(self, fileids=None):
        """Retrieve a list of lemmatised words. Words are encoded as tuples in
           C{(word, lemma)} form.
        @return: A list of words and their tuples.
        @rtype: C{list} of C{(word, lemma)}
        """

        self._require(self.WORDS, self.LEMMA)
        return LazyConcatenation(LazyMap(self._get_lemmatised_words,
                                         self._grids(fileids)))

    def lemmatised_sents(self, fileids=None):
        """Retrieve a list of sentences and the words' lemma. Words
           are encoded as tuples in C{(word, lemma)} form.
        @return: A list of sentences with words and their lemma.
        @rtype: C{list} of C{list} of C{(word, lemma)}
        """

        self._require(self.WORDS, self.LEMMA)
        return LazyMap(self._get_lemmatised_words, self._grids(fileids))

    def morphological_words(self, fileids=None):
        """Retrieve a list of sentences with the words' morphological type.
           Words are encoded as tuples in C{(word, morph)} form.
        @return: A list of sentences with words and their morphological type.
        @rtype: C{list} of C{(word, morph)}
        """

        self._require(self.WORDS, self.MORPH)
        return LazyConcatenation(LazyMap(self._get_morphological_words,
                                         self._grids(fileids)))

    def morphological_sents(self, fileids=None):
        """Retrieve a list of sentences with the words' morphological type.
           Words are encoded as tuples in C{(word, morph)} form.
        @return: A list of sentences with words and their morphological type.
        @rtype: C{list} of C{list} of C{(word, morph)}
        """

        self._require(self.WORDS, self.MORPH)
        return LazyMap(self._get_morphological_words, self._grids(fileids))

    def parsed_sents(self, fileids=None):
        """
        Retrieve a list of parsed sents as L{Tree}. The tree
        leaves are bare strings containing the word, and are children
        to unary tree nodes containing the part of speech tag.

        @return: A list of sentence tree representations.
        @rtype: C{list} of L{Tree}
        """

        self._require(self.WORDS, self.POS, self.PARENT)
        return LazyMap(self._get_parsed_words, self._grids(fileids))

    def parsed_sents_morph(self, fileids=None):
        """
        Retrieve a list of parsed sents as L{ParentedTree} with
        morphological information stored in the tree leaves. The tree
        leaves are objects which act like bare strings containing the
        word, but with extra properties C{tag}, C{morph},
        C{grid_lineno} and C{parent}; the leaves are children to unary
        tree nodes containing the part of speech tag.

        @return: A list of sentence tree representations.
        @rtype: C{list} of L{ParentedTree}
        """

        self._require(self.WORDS, self.POS, self.PARENT, self.MORPH)
        return LazyMap(self._get_parsed_words_morph, self._grids(fileids))

    #==========================================================================
    # Transforms
    #==========================================================================


    def _get_morphological_words(self, grid):
        """Retrieve the words and their morphological type.
        @return: Return a list of words and their morphological type.
        @rtype: C{list} of C{(word, morph)}
        """

        return zip(self._get_column(grid, self._colmap[self.WORDS]),
                   self._get_column(grid, self._colmap[self.MORPH]))

    def _get_lemmatised_words(self, grid):
        """Retrieve the words and their corresponding lemma.
        @return: Return a list of lemmatised words.
        @rtype: C{list} of C{(word, lemma)}
        """

        return zip(self._get_column(grid, self._colmap[self.WORDS]),
                   self._get_column(grid, self._colmap[self.LEMMA]))


    def _get_parsed_words_helper(self, tokens, node_class, node_builder):
        """
        Builds a parse tree of type C{node_class} from the grid. The
        tree leaves are built by the function node_builder, which
        takes as parameters an integer line number, a list of strings
        representing the token, and a pointer to the tree node above
        the leaf, containing the part of speech tag.

        @return: Return a tree representation of parsed words from
        the grid.
        @rtype: L{node_class}
        """

        # Build a dictionary from the tree nodes. Tree nodes are found at the
        # end of the grid. Their word column consists out of a number starting
        # with the # character identifying the node.
        nodes = dict()
        node_parents = dict()
        top_node = None
        for lineno, token in [node for node in reversed(list(enumerate(tokens)))
                              if node[1][0][0].startswith('#')]:
            word = int(token[0][1:])
            tag = token[1]
            parent = int(token[2])

            # The root node can be found at the end of the grid.
            if top_node is None and parent is 0:
                top_node = word

            # Prevents two tree roots.
            if top_node is not None and parent is 0:
                parent = top_node

            nodes[word] = node_class(tag, [])
            nodes[word].grid_lineno = lineno
            node_parents[word] = parent

        # Sentence is not correctly formeatted.
        if top_node is None:
            return None

        # Walk through the leaves and add them to their parents.
        last_parent = None
        for lineno, token in enumerate(tokens[: - len(nodes)]):
            parent = int(token[2])

            # The Negra corpus format allows tokens outside the sentence tree.
            # Prevent this, by changing their parent to the top_node's number.
            if parent is 0:
                parent = top_node

            # A chunk ends as soon as the current token has a new parent. The
            # last chunk has to be added to its own parents until it is located
            # in a subtree of the sentence tree root.
            if not parent == last_parent and last_parent is not None:
                node = last_parent
                while not node == top_node:
                    node_parent = node_parents[node]
                    if nodes[node] not in nodes[node_parent]:
                        nodes[node_parent].append(nodes[node])
                    node = node_parent

            # Add the current token to its parent.
            tag = token[1]
            node = node_class(tag, [])
            node.append(node_builder(lineno, token, node))
            node.grid_lineno = lineno
            nodes[parent].append(node)
            last_parent = parent

        return nodes[top_node]

    def _get_parsed_words(self, grid):
        """
        Builds a parse tree of type C{Tree} from the grid. The tree
        leaves are bare strings containing the word, and are children
        to unary tree nodes containing the part of speech tag.

        @return: Return a tree representation of parsed words from the grid.
        @rtype: L{Tree}
        """

        # Get the needed columns. The parent column is crucial and contains the
        # token's parent node.
        tokens = zip(
            self._get_column(grid, self._colmap[self.WORDS], filter=False),
            self._get_column(grid, self._colmap[self.POS], filter=False),
            self._get_column(grid, self._colmap[self.PARENT], filter=False)
        )

        return self._get_parsed_words_helper(tokens,
                                             Tree,
                                             lambda l, t, n: t[0])

    def _get_parsed_words_morph(self, grid):
        """
        Builds a parse tree of type C{ParentedTree} from the grid. The
        tree leaves are objects which act like bare strings containing
        the word, but with extra properties C{tag}, C{morph},
        C{grid_lineno} and C{parent}; the leaves are children to unary
        tree nodes containing the part of speech tag.

        @return: Return a tree representation of parsed words from the grid.
        @rtype: L{ParentedTree}
        """

        # Get the needed columns. The parent column is crucial and contains the
        # token's parent node.
        tokens = zip(
            self._get_column(grid, self._colmap[self.WORDS], filter=False),
            self._get_column(grid, self._colmap[self.POS], filter=False),
            self._get_column(grid, self._colmap[self.PARENT], filter=False),
            self._get_column(grid, self._colmap[self.MORPH], filter=False)
            )

        return self._get_parsed_words_helper(tokens,
                                             ParentedTree,
                                             lambda l, t, n: Atom(word=t[0],
                                                                  tag=t[1],
                                                                  morph=t[3],
                                                                  grid_lineno=l,
                                                                  parent=n))


    #==========================================================================
    # Grid reading
    #==========================================================================

    def _read_grid_block(self, stream):
        """Read blocks and return the grid"""

        # Sentence blocks are enclosed in start- and end-of-sentence tags.
        grids = []
        for block in read_regexp_block(stream, self._bos, self._eos):
            block = block.strip()
            if not block:
                continue

            # columns are separated by whitespace.
            grids.append([line.split() for line in block.split("\n")[1:]])

        return grids

    #==========================================================================
    # Helper methods
    #==========================================================================

    @staticmethod
    def _get_column(grid, column_index, filter=True):
        """Overridden; allows filtering sentence tree nodes from the grid"""

        # collect the column
        column_values = [grid[i][column_index] for i in range(len(grid))]

        # filter the column if needed
        if filter:
            column_values = [token for token in column_values
                             if token[0] is not '#']
        return column_values
