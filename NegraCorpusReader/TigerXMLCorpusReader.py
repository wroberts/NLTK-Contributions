#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
TigerXMLCorpusReader.py
(c) 14 June, 2013  Will Roberts

Read TIGER corpus files in XML format.
'''

from nltk.corpus.reader.util    import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from nltk.tree                  import Tree, ParentedTree
from nltk.util                  import LazyConcatenation, LazyMap
from NegraCorpusReader          import Atom

class TigerXMLCorpusReader(XMLCorpusReader):
    '''
    Corpus reader for the TIGER XML corpus.
    '''

    def __init__(self, root, fileids):
        '''
        Creates a new TIGER XML corpus reader.

        Arguments:
        - `root`: the base directory for the TIGER corpus
        - `fileids`: the XML filename of the TIGER corpus
        '''
        super(TigerXMLCorpusReader, self).__init__(root, fileids)

    #==========================================================================
    # Data access methods
    #==========================================================================

    def words(self, fileids=None):
        '''
        Returns all of the words and punctuation symbols that were in
        text nodes.
        '''
        return LazyConcatenation(LazyMap(self._get_words,
                                         self._sentence_etrees(fileids)))

    def sents(self, fileids=None):
        '''
        Retrieves a list of unannotated sentences from the
        corpus.
        '''
        return LazyMap(self._get_words, self._sentence_etrees(fileids))

    def tagged_words(self, fileids=None):
        return LazyConcatenation(LazyMap(self._get_tagged_words,
                                         self._sentence_etrees(fileids)))

    def tagged_sents(self, fileids=None):
        return LazyMap(self._get_tagged_words, self._sentence_etrees(fileids))

    def lemmatised_words(self, fileids=None):
        '''
        Retrieve a list of lemmatised words. Words are encoded as
        tuples in C{(word, lemma)} form.

        @return: A list of words and their tuples.
        @rtype: C{list} of C{(word, lemma)}
        '''
        return LazyConcatenation(LazyMap(self._get_lemmatised_words,
                                         self._sentence_etrees(fileids)))

    def lemmatised_sents(self, fileids=None):
        '''
        Retrieve a list of sentences and the words' lemma. Words are
        encoded as tuples in C{(word, lemma)} form.

        @return: A list of sentences with words and their lemma.
        @rtype: C{list} of C{list} of C{(word, lemma)}
        '''
        return LazyMap(self._get_lemmatised_words,
                       self._sentence_etrees(fileids))

    def morphological_words(self, fileids=None):
        '''
        Retrieve a list of sentences with the words' morphological
        type.  Words are encoded as tuples in C{(word, morph)} form.

        @return: A list of sentences with words and their morphological type.
        @rtype: C{list} of C{(word, morph)}
        '''
        return LazyConcatenation(LazyMap(self._get_morphological_words,
                                         self._sentence_etrees(fileids)))

    def morphological_sents(self, fileids=None):
        '''
        Retrieve a list of sentences with the words' morphological
        type. Words are encoded as tuples in C{(word, morph)} form.

        @return: A list of sentences with words and their morphological type.
        @rtype: C{list} of C{list} of C{(word, morph)}
        '''
        return LazyMap(self._get_morphological_words,
                       self._sentence_etrees(fileids))

    def parsed_sents(self, fileids=None):
        '''
        Retrieve a list of parsed sents as L{Tree}. The tree
        leaves are bare strings containing the word, and are children
        to unary tree nodes containing the part of speech tag.

        @return: A list of sentence tree representations.
        @rtype: C{list} of L{Tree}
        '''
        return LazyMap(self._get_parsed_words,
                       self._sentence_etrees(fileids))

    def parsed_sents_morph(self, fileids = None, secedge_copy = True):
        '''
        Retrieve a list of parsed sents as L{ParentedTree} with
        morphological information stored in the tree leaves. The tree
        leaves are objects which act like bare strings containing the
        word, but with extra properties C{tag}, C{morph},
        C{grid_lineno} and C{parent}; the leaves are children to unary
        tree nodes containing the part of speech tag.

        @return: A list of sentence tree representations.
        @rtype: C{list} of L{ParentedTree}
        '''
        return LazyMap(lambda s: self._get_parsed_words_morph(s, secedge_copy),
                       self._sentence_etrees(fileids))

    #==========================================================================
    # Transforms
    #==========================================================================

    def _sentence_etrees(self, fileids=None):
        return concat([XMLCorpusView(fileid, '.*/s')
                       for fileid in self.abspaths(fileids)])

    def _get_lemmatised_words(self, sentence_etree):
        graph = sentence_etree.find('graph')
        return [(unicode(terminal.get('word')), unicode(terminal.get('lemma'))) for
                terminal in graph.getiterator("t")]

    def _get_morphological_words(self, sentence_etree):
        graph = sentence_etree.find('graph')
        return [(unicode(terminal.get('word')), unicode(terminal.get('morph'))) for
                terminal in graph.getiterator("t")]

    def _get_parsed_words(self, sentence_etree):
        '''
        Builds a parse tree of type C{Tree} from the grid. The tree
        leaves are bare strings containing the word, and are children
        to unary tree nodes containing the part of speech tag.

        @return: Return a tree representation of parsed words from the grid.
        @rtype: L{Tree}
        '''
        return _sentence_etree_to_tree(sentence_etree,
                                       Tree,
                                       lambda l, t, p: unicode(t.get('word')),
                                       False)

    def _get_parsed_words_morph(self, sentence_etree, secedge_copy = True):
        '''
        Builds a parse tree of type C{ParentedTree} from the grid. The
        tree leaves are objects which act like bare strings containing
        the word, but with extra properties C{tag}, C{morph},
        C{grid_lineno} and C{parent}; the leaves are children to unary
        tree nodes containing the part of speech tag.

        @return: Return a tree representation of parsed words from the grid.
        @rtype: L{ParentedTree}
        '''
        return _sentence_etree_to_tree(sentence_etree,
                                       ParentedTree,
                                       lambda l, t, p: Atom(word=unicode(t.get('word')),
                                                            tag=unicode(t.get('pos', None)),
                                                            morph=unicode(t.get('morph', None)),
                                                            lemma=unicode(t.get('lemma', None)),
                                                            edge=None,
                                                            secedge=None,
                                                            comment=None,
                                                            grid_lineno=l,
                                                            parent=p),
                                       secedge_copy)

    def _get_tagged_words(self, sentence_etree):
        graph = sentence_etree.find('graph')
        return [(unicode(terminal.get('word')), unicode(terminal.get('pos'))) for
                terminal in graph.getiterator("t")]

    def _get_words(self, sentence_etree):
        graph = sentence_etree.find('graph')
        return [unicode(terminal.get('word')) for terminal in graph.getiterator("t")]

def _copy_subtree_helper(subtree, label, parent_idref, tokens, terminal_etrees,
                         tree_class, atom_builder):
    subtree_copy             = tree_class(subtree.node, [])
    subtree_copy.grid_lineno = subtree.grid_lineno
    subtree_copy.edge        = label
    todo = []
    for child in subtree:
        todo.append((subtree_copy, child))
    while todo:
        (subtree_parent, current) = todo.pop(0)
        if isinstance(current, tree_class):
            current_copy             = tree_class(current.node, [])
            current_copy.grid_lineno = current.grid_lineno
            current_copy.edge        = current.edge
            for child in current:
                todo.append((current_copy, child))
        else:
            current_copy = atom_builder(subtree_parent.grid_lineno,
                                        terminal_etrees[subtree_parent.grid_lineno],
                                        subtree_parent)
            if subtree_parent is subtree_copy:
                if isinstance(current_copy, Atom):
                    current_copy.edge = label
        subtree_parent.append(current_copy)
    tokens[parent_idref].append(subtree_copy)

def _sentence_etree_to_tree(sentence_etree, tree_class, atom_builder,
                            secedge_copy = True):
    '''
    Helper function to transform an ElementTree element read from the
    TIGER XML corpus into an NLTK tree.
    '''
    graph           = sentence_etree.find('graph')
    vroot_id        = graph.get('root')
    skip_vroot      = ((vroot_id.split('_')[1].lower() == 'vroot') and
                       len(list(graph.getiterator('nt'))) > 1)
    tokens          = {}
    secedges        = []
    terminal_etrees = {}
    terminal_ids    = set()
    # build the list of terminals
    for idx, terminal in enumerate(graph.getiterator('t')):
        tok = tree_class(unicode(terminal.get('pos')), [])
        tok.grid_lineno = idx
        tok.edge        = None
        atom = atom_builder(idx, terminal, tok)
        tok.append(atom)
        tokens[terminal.get('id')] = tok
        terminal_ids.add(terminal.get('id'))
        terminal_etrees[idx] = terminal
        for secedge in terminal.getiterator('secedge'):
            secedges.append((tok, unicode(secedge.get('label')),
                             secedge.get('idref')))
    num_terminals = len(tokens)
    root_id       = (None if skip_vroot else vroot_id)
    # build the list of non-terminals
    for idx, nonterminal in enumerate(graph.getiterator('nt')):
        idx += num_terminals
        if not (nonterminal.get('id') == vroot_id and skip_vroot):
            tok = tree_class(unicode(nonterminal.get('cat')), [])
            tok.grid_lineno = idx
            tok.edge        = None
            tokens[nonterminal.get('id')] = tok
            for secedge in nonterminal.getiterator('secedge'):
                secedges.append((tok, unicode(secedge.get('label')),
                                 secedge.get('idref')))
        else:
            for edge in nonterminal.getiterator('edge'):
                if edge.get('idref') not in terminal_ids:
                    root_id = edge.get('idref')
    # attach terminals and non-terminals to their parents using the
    # information in <edge> tags
    attached_ids = set()
    for nonterminal in graph.getiterator('nt'):
        if not (nonterminal.get('id') == vroot_id and skip_vroot):
            tok = tokens[nonterminal.get('id')]
            for edge in nonterminal.getiterator('edge'):
                # we can't attach the same constituent to two
                # different parents
                if edge.get('idref') in attached_ids:
                    return None
                attached_ids.add(edge.get('idref'))
                child = tokens[edge.get('idref')]
                child.edge = unicode(edge.get('label'))
                if (isinstance(child, tree_class) and
                    len(child) == 1 and
                    isinstance(child[0], Atom)):
                    child[0].edge = unicode(edge.get('label'))
                tok.append(child)
        else:
            for edge in nonterminal.getiterator('edge'):
                if edge.get('idref') != root_id:
                    child = tokens[edge.get('idref')]
                    child.edge = unicode(edge.get('label'))
                    if (isinstance(child, tree_class) and
                        len(child) == 1 and
                        isinstance(child[0], Atom)):
                        child[0].edge = unicode(edge.get('label'))
                    tokens[root_id].append(child)
    # process secedges
    if secedge_copy:
        for (subtree, label, parent_idref) in secedges:
            _copy_subtree_helper(subtree, label, parent_idref, tokens,
                                 terminal_etrees, tree_class, atom_builder)
    return tokens[root_id]
