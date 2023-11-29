// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Shanghai Anlogic Infotech Co.,Ltd.
// Copyright (c) 2023-2025 Peking University
//
// iMAP-FPGA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************

#pragma once
#include "details/node.hpp"
#include "details/storage.hpp"
#include "details/events.hpp"
#include "details/foreach.hpp"

#include "utils/traits.hpp"
#include "utils/range.hpp"
#include "utils/truth_table_cache.hpp"
#include "kitty/dynamic_truth_table.hpp"
#include "kitty/constructors.hpp"

iFPGA_NAMESPACE_HEADER_START

struct klut_storage_data
{
  truth_table_cache< kitty::dynamic_truth_table > cache;
  uint32_t              num_pis{0u};
  uint32_t              num_pos{0u};
  std::vector<uint32_t> latches{};
  uint32_t              trav_id{0u};  
};

/**
 * @brief klut node
 *        data[0].h1: Fan-out size
 *        data[0].h2: Application-specific value
 *        data[1].h1: Function literal in truth table cache
 *        data[2].h2: Visited flags
 */
struct klut_storage_node : mixed_node<2>
{
  bool operator==( klut_storage_node const& other ) const
  {
    return data[1].h1 == other.data[1].h1 && children == other.children;
  }
};

using klut_storage = storage< klut_storage_node, klut_storage_data >;

class klut_network
{
public:
#pragma region Types and constructors
  static constexpr auto min_fanin_size = 1;
  static constexpr auto max_fanin_size = 32;

  using base_type = klut_network;
  using storage = std::shared_ptr<klut_storage>;
  using node = uint64_t;
  using signal = uint64_t;

  klut_network()
      : _storage( std::make_shared<klut_storage>() ),
        _events( std::make_shared<decltype( _events )::element_type>() )
  {
    _init();
  }

  klut_network( std::shared_ptr<klut_storage> storage )
      : _storage( storage ),
        _events( std::make_shared<decltype( _events )::element_type>() )
  { 

  }

private:
  inline void _init()
  {
    /* reserve the second node for constant 1 */
    _storage->nodes.emplace_back();

    /* reserve some truth tables for nodes */
    kitty::dynamic_truth_table tt_zero( 0 );
    _storage->data.cache.insert( tt_zero );

    static uint64_t _not = 0x1;
    kitty::dynamic_truth_table tt_not( 1 );
    kitty::create_from_words( tt_not, &_not, &_not + 1 );
    _storage->data.cache.insert( tt_not );

    static uint64_t _and = 0x8;
    kitty::dynamic_truth_table tt_and( 2 );
    kitty::create_from_words( tt_and, &_and, &_and + 1 );
    _storage->data.cache.insert( tt_and );

    static uint64_t _or = 0xe;
    kitty::dynamic_truth_table tt_or( 2 );
    kitty::create_from_words( tt_or, &_or, &_or + 1 );
    _storage->data.cache.insert( tt_or );

    static uint64_t _lt = 0x4;
    kitty::dynamic_truth_table tt_lt( 2 );
    kitty::create_from_words( tt_lt, &_lt, &_lt + 1 );
    _storage->data.cache.insert( tt_lt );

    static uint64_t _le = 0xd;
    kitty::dynamic_truth_table tt_le( 2 );
    kitty::create_from_words( tt_le, &_le, &_le + 1 );
    _storage->data.cache.insert( tt_le );

    static uint64_t _xor = 0x6;
    kitty::dynamic_truth_table tt_xor( 2 );
    kitty::create_from_words( tt_xor, &_xor, &_xor + 1 );
    _storage->data.cache.insert( tt_xor );

    static uint64_t _maj = 0xe8;
    kitty::dynamic_truth_table tt_maj( 3 );
    kitty::create_from_words( tt_maj, &_maj, &_maj + 1 );
    _storage->data.cache.insert( tt_maj );

    static uint64_t _ite = 0xd8;
    kitty::dynamic_truth_table tt_ite( 3 );
    kitty::create_from_words( tt_ite, &_ite, &_ite + 1 );
    _storage->data.cache.insert( tt_ite );

    static uint64_t _xor3 = 0x96;
    kitty::dynamic_truth_table tt_xor3( 3 );
    kitty::create_from_words( tt_xor3, &_xor3, &_xor3 + 1 );
    _storage->data.cache.insert( tt_xor3 );

    /* truth tables for constants */
    _storage->nodes[0].data[1].h1 = 0;
    _storage->nodes[1].data[1].h1 = 1;
  }
#pragma endregion

#pragma region Primary I / O and constants
public:
  signal get_constant( bool value = false ) const
  {
    return value ? 1 : 0;
  }

  signal create_pi( std::string const& name = std::string() )
  {
    (void)name;

    const auto index = _storage->nodes.size();
    _storage->nodes.emplace_back();
    _storage->inputs.emplace_back( index );
    _storage->nodes[index].data[1].h1 = 2;
    ++_storage->data.num_pis;

    _storage->nodes[index_to_node(index)].data[0].h2 = 1;
    return index;
  }

  uint32_t create_po( signal const& f, std::string const& name = std::string() )
  {
    (void)name;

    /* increase ref-count to children */
    _storage->nodes[f].data[0].h1++;

    auto const po_index = static_cast<uint32_t>( _storage->outputs.size() );
    _storage->outputs.emplace_back( f );
    ++_storage->data.num_pos;
    return po_index;
  }

  signal create_ro( std::string const& name = std::string() )
  {
    (void)name;

    auto const index = static_cast<uint32_t>( _storage->nodes.size() );
    _storage->nodes.emplace_back();
    _storage->inputs.emplace_back( index );
    _storage->nodes[index].data[1].h1 = 2;

    _storage->nodes[index_to_node(index)].data[0].h2 = 3;
    return index;
  }

  uint32_t create_ri( signal const& f, int8_t reset = 0, std::string const& name = std::string() )
  {
    (void)name;

    /* increase ref-count to children */
    _storage->nodes[f].data[0].h1++;
    auto const ri_index = static_cast<uint32_t>( _storage->outputs.size() );
    _storage->outputs.emplace_back( f );
    _storage->data.latches.emplace_back( reset );
    return ri_index;
  }

  int8_t latch_reset( uint32_t index ) const
  {
    assert( index < _storage->data.latches.size() );
    return _storage->data.latches[index];
  }

  bool is_combinational() const
  {
    return ( static_cast<uint32_t>( _storage->inputs.size() ) == _storage->data.num_pis &&
             static_cast<uint32_t>( _storage->outputs.size() ) == _storage->data.num_pos );
  }

  bool is_constant( node const& n ) const
  {
    return n <= 1;
  }

  bool is_ci( node const& n ) const
  {
    return _storage->nodes[n].data[0].h2 == 1 ||
           _storage->nodes[n].data[0].h2 == 3;
  }

  bool is_pi( node const& n ) const
  {
    return _storage->nodes[n].data[0].h2 == 1;
  }

  bool is_ro( node const& n ) const
  {
    bool found = false;
    detail::foreach_element( _storage->inputs.begin() + _storage->data.num_pis, _storage->inputs.end(), [&found,&n]( auto const& node ){
        if ( node == n )
        {
          found = true;
          return false;
        }
        return true;
      } );
    return found;
  }

  bool constant_value( node const& n ) const
  {
    return n == 1;
  }
#pragma endregion

#pragma region Create unary functions
  signal create_buf( signal const& a )
  {
    return a;
  }

  signal create_not( signal const& a )
  {
    return _create_node( {a}, 3 );
  }
#pragma endregion

#pragma region Create binary functions
  signal create_and( signal a, signal b )
  {
    return _create_node( {a, b}, 4 );
  }

  signal create_or( signal a, signal b )
  {
    return _create_node( {a, b}, 6 );
  }

  signal create_lt( signal a, signal b )
  {
    return _create_node( {a, b}, 8 );
  }

  signal create_le( signal a, signal b )
  {
    return _create_node( {a, b}, 11 );
  }

  signal create_xor( signal a, signal b )
  {
    return _create_node( {a, b}, 12 );
  }
#pragma endregion

#pragma region Create ternary functions
signal create_maj( signal a, signal b, signal c )
  {
    return _create_node( {a, b, c}, 14 );
  }

  signal create_ite( signal a, signal b, signal c )
  {
    return _create_node( {a, b, c}, 16 );
  }

  signal create_xor3( signal a, signal b, signal c )
  {
    return _create_node( {a, b, c}, 18 );
  }
#pragma endregion

#pragma region Create nary functions
  signal create_nary_and( std::vector<signal> const& fs )
  {
    return tree_reduce( fs.begin(), fs.end(), get_constant( true ), [this]( auto const& a, auto const& b ) { return create_and( a, b ); } );
  }

  signal create_nary_or( std::vector<signal> const& fs )
  {
    return tree_reduce( fs.begin(), fs.end(), get_constant( false ), [this]( auto const& a, auto const& b ) { return create_or( a, b ); } );
  }

  signal create_nary_xor( std::vector<signal> const& fs )
  {
    return tree_reduce( fs.begin(), fs.end(), get_constant( false ), [this]( auto const& a, auto const& b ) { return create_xor( a, b ); } );
  }
#pragma endregion

#pragma region Create arbitrary functions
  signal _create_node( std::vector<signal> const& children, uint32_t literal )
  {
    storage::element_type::node_type node;
    std::copy( children.begin(), children.end(), std::back_inserter( node.children ) );
    node.data[1].h1 = literal;

    //FIXME, skip hash
    //const auto it = _storage->hash.find( node );
    //if ( it != _storage->hash.end() )
    //{
    //  return it->second;
    //}

    const auto index = _storage->nodes.size();
    _storage->nodes.push_back( node );
    //_storage->hash[node] = index;

    /* increase ref-count to children */
    for ( auto c : children )
    {
      _storage->nodes[c].data[0].h1++;
    }

    set_value( index, 0 );

    for ( auto const& fn : _events->on_add )
    {
      fn( index );
    }

    return index;
  }

  signal create_node( std::vector<signal> const& children, kitty::dynamic_truth_table const& function )
  {
    if ( children.size() == 0u )
    {
      assert( function.num_vars() == 0u );
      return get_constant( !kitty::is_const0( function ) );
    }
    return _create_node( children, _storage->data.cache.insert( function ) );
  }

  signal clone_node( klut_network const& other, node const& source, std::vector<signal> const& children )
  {
    assert( !children.empty() );
    const auto tt = other._storage->data.cache[other._storage->nodes[source].data[1].h1];
    return create_node( children, tt );
  }
#pragma endregion

#pragma region Restructuring
  void substitute_node( node const& old_node, signal const& new_signal )
  {
    /* find all parents from old_node */
    for ( auto i = 0u; i < _storage->nodes.size(); ++i )
    {
      auto& n = _storage->nodes[i];
      for ( auto& child : n.children )
      {
        if ( child == old_node )
        {
          std::vector<signal> old_children( n.children.size() );
          std::transform( n.children.begin(), n.children.end(), old_children.begin(), []( auto c ) { return c.index; } );
          child = new_signal;

          // increment fan-out of new node
          _storage->nodes[new_signal].data[0].h1++;

          for ( auto const& fn : _events->on_modified )
          {
            fn( i, old_children );
          }
        }
      }
    }

    /* check outputs */
    for ( auto& output : _storage->outputs )
    {
      if ( output == old_node )
      {
        output = new_signal;

        // increment fan-out of new node
        _storage->nodes[new_signal].data[0].h1++;
      }
    }

    // reset fan-out of old node
    _storage->nodes[old_node].data[0].h1 = 0;
  }
#pragma endregion

#pragma region Structural properties
  auto size() const
  {
    return static_cast<uint32_t>( _storage->nodes.size() );
  }

  auto num_cis() const
  {
    return static_cast<uint32_t>( _storage->inputs.size() );
  }

  auto num_cos() const
  {
    return static_cast<uint32_t>( _storage->outputs.size() );
  }

  uint32_t num_latches() const
  {
      return static_cast<uint32_t>( _storage->data.latches.size() );
  }

  auto num_pis() const
  {
    return _storage->data.num_pis;
  }

  auto num_pos() const
  {
    return _storage->data.num_pos;
  }

  auto num_registers() const
  {
    assert( static_cast<uint32_t>( _storage->inputs.size() - _storage->data.num_pis ) == static_cast<uint32_t>( _storage->outputs.size() - _storage->data.num_pos ) );
    return static_cast<uint32_t>( _storage->inputs.size() - _storage->data.num_pis );
  }

  auto num_gates() const
  {
    return static_cast<uint32_t>(_storage->nodes.size() - _storage->inputs.size() - 2 );
  }

  uint32_t fanin_size( node const& n ) const
  {
    return static_cast<uint32_t>( _storage->nodes[n].children.size() );
  }

  uint32_t fanout_size( node const& n ) const
  {
    return _storage->nodes[n].data[0].h1;
  }

  bool is_function( node const& n ) const
  {
    return n > 1 && !is_ci( n );
  }
#pragma endregion

#pragma region Functional properties
  kitty::dynamic_truth_table node_function( const node& n ) const
  {
    return _storage->data.cache[_storage->nodes[n].data[1].h1];
  }
#pragma endregion

#pragma region Nodes and signals
  node get_node( signal const& f ) const
  {
    return f;
  }

  signal make_signal( node const& n ) const
  {
    return n;
  }

  bool is_complemented( signal const& f ) const
  {
    (void)f;
    return false;
  }

  uint32_t node_to_index( node const& n ) const
  {
    return static_cast<uint32_t>( n );
  }

  node index_to_node( uint32_t index ) const
  {
    return index;
  }

  node ci_at( uint32_t index ) const
  {
    assert( index < _storage->inputs.size() );
    return *(_storage->inputs.begin() + index);
  }

  signal co_at( uint32_t index ) const
  {
    assert( index < _storage->outputs.size() );
    return (_storage->outputs.begin() + index)->index;
  }

  node pi_at( uint32_t index ) const
  {
    assert( index < _storage->data.num_pis );
    return *(_storage->inputs.begin() + index);
  }

  signal po_at( uint32_t index ) const
  {
    assert( index < _storage->data.num_pos );
    return (_storage->outputs.begin() + index)->index;
  }

  node ro_at( uint32_t index ) const
  {
    assert( index < _storage->inputs.size() - _storage->data.num_pis );
    return *(_storage->inputs.begin() + _storage->data.num_pis + index);
  }

  signal ri_at( uint32_t index ) const
  {
    assert( index < _storage->outputs.size() - _storage->data.num_pos );
    return (_storage->outputs.begin() + _storage->data.num_pos + index)->index;
  }

  uint32_t ci_index( node const& n ) const
  {
    assert( _storage->nodes[n].children[0].data == _storage->nodes[n].children[1].data );
    return static_cast<uint32_t>( _storage->nodes[n].children[0].data );
  }

  uint32_t co_index( signal const& s ) const
  {
    uint32_t i = -1;
    foreach_co( [&]( const auto& x, auto index ){
        if ( x == s )
        {
          i = index;
          return false;
        }
        return true;
      });
    return i;
  }

  uint32_t pi_index( node const& n ) const
  {
    assert( _storage->nodes[n].children[0].data == _storage->nodes[n].children[1].data );
    return static_cast<uint32_t>( _storage->nodes[n].children[0].data );
  }

  uint32_t po_index( signal const& s ) const
  {
    uint32_t i = -1;
    foreach_po( [&]( const auto& x, auto index ){
        if ( x == s )
        {
          i = index;
          return false;
        }
        return true;
      });
    return i;
  }

  uint32_t ro_index( node const& n ) const
  {
    assert( _storage->nodes[n].children[0].data == _storage->nodes[n].children[1].data );
    return static_cast<uint32_t>( _storage->nodes[n].children[0].data - _storage->data.num_pis );
  }

  uint32_t ri_index( signal const& s ) const
  {
    uint32_t i = -1;
    foreach_ri( [&]( const auto& x, auto index ){
        if ( x == s )
        {
          i = index;
          return false;
        }
        return true;
      });
    return i;
  }

  signal ro_to_ri( signal const& s ) const
  {
    return ( _storage->outputs.begin() + _storage->data.num_pos + _storage->nodes[s].children[0].data - _storage->data.num_pis )->index;
  }

  node ri_to_ro( signal const& s ) const
  {
    return *( _storage->inputs.begin() + _storage->data.num_pis + ri_index( s ) );
  }
#pragma endregion

#pragma region Node and signal iterators
  template<typename Fn>
  void foreach_node( Fn&& fn ) const
  {
    auto r = range<uint64_t>( _storage->nodes.size() );
    detail::foreach_element( r.begin(), r.end(), fn );
  }

  template<typename Fn>
  void foreach_ci( Fn&& fn ) const
  {
    detail::foreach_element( _storage->inputs.begin(), _storage->inputs.end(), fn );
  }

  template<typename Fn>
  void foreach_co( Fn&& fn ) const
  {
    using IteratorType = decltype( _storage->outputs.begin() );
    detail::foreach_element_transform<IteratorType, uint32_t>( _storage->outputs.begin(), _storage->outputs.end(), []( auto o ) { return o.index; }, fn );
  }

  template<typename Fn>
  void foreach_pi( Fn&& fn ) const
  {
    detail::foreach_element( _storage->inputs.begin(), _storage->inputs.begin() + _storage->data.num_pis, fn );
  }

  template<typename Fn>
  void foreach_po( Fn&& fn ) const
  {
    using IteratorType = decltype( _storage->outputs.begin() );
    detail::foreach_element_transform<IteratorType, uint32_t>( _storage->outputs.begin(), _storage->outputs.begin() + _storage->data.num_pos, []( auto o ) { return o.index; }, fn );
  }

  template<typename Fn>
  void foreach_ro( Fn&& fn ) const
  {
    detail::foreach_element( _storage->inputs.begin() + _storage->data.num_pis, _storage->inputs.end(), fn );
  }

  template<typename Fn>
  void foreach_ri( Fn&& fn ) const
  {
    using IteratorType = decltype( _storage->outputs.begin() );
    detail::foreach_element_transform<IteratorType, uint32_t>( _storage->outputs.begin() + _storage->data.num_pos, _storage->outputs.end(), []( auto o ) { return o.index; }, fn );
  }

  template<typename Fn>
  void foreach_register( Fn&& fn ) const
  {
    static_assert( detail::is_callable_with_index_v<Fn, std::pair<signal,node>, void> ||
                   detail::is_callable_without_index_v<Fn, std::pair<signal,node>, void> ||
                   detail::is_callable_with_index_v<Fn, std::pair<signal,node>, bool> ||
                   detail::is_callable_without_index_v<Fn, std::pair<signal,node>, bool> );

    assert( _storage->inputs.size() - _storage->data.num_pis == _storage->outputs.size() - _storage->data.num_pos );
    auto ro = _storage->inputs.begin() + _storage->data.num_pis;
    auto ri = _storage->outputs.begin() + _storage->data.num_pos;
    if constexpr ( detail::is_callable_without_index_v<Fn, std::pair<signal,node>, bool> )
    {
      while ( ro != _storage->inputs.end() && ri != _storage->outputs.end() )
      {
        if ( !fn( std::make_pair( ( ri++ )->index, ro++ ) ) )
          return;
      }
    }
    else if constexpr ( detail::is_callable_with_index_v<Fn, std::pair<signal,node>, bool> )
    {
      uint32_t index{0};
      while ( ro != _storage->inputs.end() && ri != _storage->outputs.end() )
      {
        if ( !fn( std::make_pair( ( ri++ )->index, ro++ ), index++ ) )
          return;
      }
    }
    else if constexpr( detail::is_callable_without_index_v<Fn, std::pair<signal,node>, void> )
    {
      while ( ro != _storage->inputs.end() && ri != _storage->outputs.end() )
      {
        fn( std::make_pair( ( ri++ )->index, *ro++ ) );
      }
    }
    else if constexpr ( detail::is_callable_with_index_v<Fn, std::pair<signal,node>, void> )
    {
      uint32_t index{0};
      while ( ro != _storage->inputs.end() && ri != _storage->outputs.end() )
      {
        fn( std::make_pair( ( ri++ )->index, *ro++ ), index++ );
      }
    }
  }

  template<typename Fn>
  void foreach_gate( Fn&& fn ) const
  {
    auto r = range<uint64_t>( 2u, _storage->nodes.size() ); /* start from 2 to avoid constants */
    detail::foreach_element_if( r.begin(), r.end(),
                                [this]( auto n ) { return !is_ci( n ); },
                                fn );
  }

  template<typename Fn>
  void foreach_fanin( node const& n, Fn&& fn ) const
  {
    if ( n == 0 || is_ci( n ) )
      return;

    using IteratorType = decltype( _storage->outputs.begin() );
    detail::foreach_element_transform<IteratorType, uint32_t>( _storage->nodes[n].children.begin(), _storage->nodes[n].children.end(), []( auto f ) { return f.index; }, fn );
  }
#pragma endregion

#pragma region Simulate values
  template<typename Iterator>
  iterates_over_t<Iterator, bool>
  compute( node const& n, Iterator begin, Iterator end ) const
  {
    uint32_t index{0};
    while ( begin != end )
    {
      index <<= 1;
      index ^= *begin++ ? 1 : 0;
    }
    return kitty::get_bit( _storage->data.cache[_storage->nodes[n].data[1].h1], index );
  }

  template<typename Iterator>
  iterates_over_truth_table_t<Iterator>
  compute( node const& n, Iterator begin, Iterator end ) const
  {
    const auto nfanin = _storage->nodes[n].children.size();
    std::vector<typename Iterator::value_type> tts( begin, end );

    assert( nfanin != 0 );
    assert( tts.size() == nfanin );

    /* resulting truth table has the same size as any of the children */
    auto result = tts.front().construct();
    const auto gate_tt = _storage->data.cache[_storage->nodes[n].data[1].h1];

    for ( uint32_t i = 0u; i < static_cast<uint32_t>( result.num_bits() ); ++i )
    {
      uint32_t pattern = 0u;
      for ( auto j = 0u; j < nfanin; ++j )
      {
        pattern |= kitty::get_bit( tts[j], i ) << j;
      }
      if ( kitty::get_bit( gate_tt, pattern ) )
      {
        kitty::set_bit( result, i );
      }
    }

    return result;
  }
#pragma endregion

#pragma region Custom node values
  void clear_values() const
  {
    std::for_each( _storage->nodes.begin(), _storage->nodes.end(), []( auto& n ) { n.data[0].h2 = 0; } );
  }

  uint32_t value( node const& n ) const
  {
    return _storage->nodes[n].data[0].h2;
  }

  void set_value( node const& n, uint32_t v ) const
  {
    _storage->nodes[n].data[0].h2 = v;
  }

  uint32_t incr_value( node const& n ) const
  {
    return static_cast<uint32_t>( _storage->nodes[n].data[0].h2++ );
  }

  uint32_t decr_value( node const& n ) const
  {
    return static_cast<uint32_t>( --_storage->nodes[n].data[0].h2 );
  }
#pragma endregion

#pragma region Visited flags
  void clear_visited() const
  {
    std::for_each( _storage->nodes.begin(), _storage->nodes.end(), []( auto& n ) { n.data[1].h2 = 0; } );
  }

  auto visited( node const& n ) const
  {
    return _storage->nodes[n].data[1].h2;
  }

  void set_visited( node const& n, uint32_t v ) const
  {
    _storage->nodes[n].data[1].h2 = v;
  }

  uint32_t trav_id() const
  {
    return _storage->data.trav_id;
  }

  void incr_trav_id() const
  {
    ++_storage->data.trav_id;
  }
#pragma endregion

#pragma region General methods
  auto& events() const
  {
    return *_events;
  }
#pragma endregion

  std::vector<uint32_t> get_detailed_lut_statics(uint32_t k) const
  {
    std::vector<uint32_t> lut_num(k, 0);
    foreach_gate( [&]( auto const& n){
        // uint32_t count_child = this->fanin_size( n ) - 1; 
        ++lut_num[ fanin_size( n ) - 1 ];
    });
    return lut_num;
  }

  std::vector<uint32_t> get_Statics_LUT(uint32_t k)  const
  {
    return get_detailed_lut_statics(k);
  }

public:
  std::shared_ptr<klut_storage> _storage;
  std::shared_ptr<network_events<base_type>> _events;
};

iFPGA_NAMESPACE_HEADER_END