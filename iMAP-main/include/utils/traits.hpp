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

#include <string>
#include <type_traits>
#include <list>
#include <map>

#include <kitty/dynamic_truth_table.hpp>
#include <kitty/traits.hpp>

#include "ifpga_namespaces.hpp"
iFPGA_NAMESPACE_HEADER_START

template<typename Ntk>
using signal = typename Ntk::signal;

template<typename Ntk>
using node = typename Ntk::node;

template<class Ntk, class = void>
struct is_network_type : std::false_type
{
};

template<class Ntk>
struct is_network_type<Ntk, std::enable_if_t<
                                std::is_constructible_v<signal<Ntk>, node<Ntk>>,
                                std::void_t<typename Ntk::base_type,
                                            signal<Ntk>,
                                            node<Ntk>,
                                            typename Ntk::storage,
                                            decltype( Ntk::max_fanin_size ),
                                            decltype( Ntk::min_fanin_size )>>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool is_network_type_v = is_network_type<Ntk>::value;

#pragma region is_topologically_sorted
template<class Ntk, class = void>
struct is_topologically_sorted : std::false_type
{
};

template<class Ntk>
struct is_topologically_sorted<Ntk, std::enable_if_t<Ntk::is_topologically_sorted, std::void_t<decltype( Ntk::is_topologically_sorted )>>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool is_topologically_sorted_v = is_topologically_sorted<Ntk>::value;
#pragma endregion

#pragma region has_get_constant
template<class Ntk, class = void>
struct has_get_constant : std::false_type
{
};

template<class Ntk>
struct has_get_constant<Ntk, std::void_t<decltype( std::declval<Ntk>().get_constant( bool() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_get_constant_v = has_get_constant<Ntk>::value;
#pragma endregion

#pragma region has_create_pi
template<class Ntk, class = void>
struct has_create_pi : std::false_type
{
};

template<class Ntk>
struct has_create_pi<Ntk, std::void_t<decltype( std::declval<Ntk>().create_pi( std::string() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_pi_v = has_create_pi<Ntk>::value;
#pragma endregion

#pragma region has_create_po
template<class Ntk, class = void>
struct has_create_po : std::false_type
{
};

template<class Ntk>
struct has_create_po<Ntk, std::void_t<decltype( std::declval<Ntk>().create_po( std::declval<signal<Ntk>>(), std::string() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_po_v = has_create_po<Ntk>::value;
#pragma endregion

#pragma region has_create_ro
template<class Ntk, class = void>
struct has_create_ro : std::false_type
{
};

template<class Ntk>
struct has_create_ro<Ntk, std::void_t<decltype( std::declval<Ntk>().create_ro( std::string() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_ro_v = has_create_ro<Ntk>::value;
#pragma endregion

#pragma region has_create_ri
template<class Ntk, class = void>
struct has_create_ri : std::false_type
{
};

template<class Ntk>
struct has_create_ri<Ntk, std::void_t<decltype( std::declval<Ntk>().create_ri( std::declval<signal<Ntk>>(), int8_t(), std::string() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_ri_v = has_create_ri<Ntk>::value;
#pragma endregion

#pragma region has_is_combinational
template<class Ntk, class = void>
struct has_is_combinational : std::false_type
{
};

template<class Ntk>
struct has_is_combinational<Ntk, std::void_t<decltype( std::declval<Ntk>().is_combinational() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_combinational_v = has_is_combinational<Ntk>::value;
#pragma endregion

#pragma region has_is_constant
template<class Ntk, class = void>
struct has_is_constant : std::false_type
{
};

template<class Ntk>
struct has_is_constant<Ntk, std::void_t<decltype( std::declval<Ntk>().is_constant( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_constant_v = has_is_constant<Ntk>::value;
#pragma endregion

#pragma region has_is_ci
template<class Ntk, class = void>
struct has_is_ci : std::false_type
{
};

template<class Ntk>
struct has_is_ci<Ntk, std::void_t<decltype( std::declval<Ntk>().is_ci( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_ci_v = has_is_ci<Ntk>::value;
#pragma endregion

#pragma region has_is_pi
template<class Ntk, class = void>
struct has_is_pi : std::false_type
{
};

template<class Ntk>
struct has_is_pi<Ntk, std::void_t<decltype( std::declval<Ntk>().is_pi( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_pi_v = has_is_pi<Ntk>::value;
#pragma endregion

#pragma region has_is_ro
template<class Ntk, class = void>
struct has_is_ro : std::false_type
{
};

template<class Ntk>
struct has_is_ro<Ntk, std::void_t<decltype( std::declval<Ntk>().is_ro( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_ro_v = has_is_ro<Ntk>::value;
#pragma endregion

#pragma region has_constant_value
template<class Ntk, class = void>
struct has_constant_value : std::false_type
{
};

template<class Ntk>
struct has_constant_value<Ntk, std::void_t<decltype( std::declval<Ntk>().constant_value( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_constant_value_v = has_constant_value<Ntk>::value;
#pragma endregion

#pragma region has_create_buf
template<class Ntk, class = void>
struct has_create_buf : std::false_type
{
};

template<class Ntk>
struct has_create_buf<Ntk, std::void_t<decltype( std::declval<Ntk>().create_buf( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_buf_v = has_create_buf<Ntk>::value;
#pragma endregion

#pragma region has_create_not
template<class Ntk, class = void>
struct has_create_not : std::false_type
{
};

template<class Ntk>
struct has_create_not<Ntk, std::void_t<decltype( std::declval<Ntk>().create_not( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_not_v = has_create_not<Ntk>::value;
#pragma endregion

#pragma region has_create_and
template<class Ntk, class = void>
struct has_create_and : std::false_type
{
};

template<class Ntk>
struct has_create_and<Ntk, std::void_t<decltype( std::declval<Ntk>().create_and( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_and_v = has_create_and<Ntk>::value;
#pragma endregion

#pragma region has_create_nand
template<class Ntk, class = void>
struct has_create_nand : std::false_type
{
};

template<class Ntk>
struct has_create_nand<Ntk, std::void_t<decltype( std::declval<Ntk>().create_nand( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_nand_v = has_create_nand<Ntk>::value;
#pragma endregion

#pragma region has_create_or
template<class Ntk, class = void>
struct has_create_or : std::false_type
{
};

template<class Ntk>
struct has_create_or<Ntk, std::void_t<decltype( std::declval<Ntk>().create_or( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_or_v = has_create_or<Ntk>::value;
#pragma endregion

#pragma region has_create_nor
template<class Ntk, class = void>
struct has_create_nor : std::false_type
{
};

template<class Ntk>
struct has_create_nor<Ntk, std::void_t<decltype( std::declval<Ntk>().create_nor( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_nor_v = has_create_nor<Ntk>::value;
#pragma endregion

#pragma region has_create_lt
template<class Ntk, class = void>
struct has_create_lt : std::false_type
{
};

template<class Ntk>
struct has_create_lt<Ntk, std::void_t<decltype( std::declval<Ntk>().create_lt( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_lt_v = has_create_lt<Ntk>::value;
#pragma endregion

#pragma region has_create_le
template<class Ntk, class = void>
struct has_create_le : std::false_type
{
};

template<class Ntk>
struct has_create_le<Ntk, std::void_t<decltype( std::declval<Ntk>().create_le( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_le_v = has_create_le<Ntk>::value;
#pragma endregion

#pragma region has_create_gt
template<class Ntk, class = void>
struct has_create_gt : std::false_type
{
};

template<class Ntk>
struct has_create_gt<Ntk, std::void_t<decltype( std::declval<Ntk>().create_gt( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_gt_v = has_create_gt<Ntk>::value;
#pragma endregion

#pragma region has_create_ge
template<class Ntk, class = void>
struct has_create_ge : std::false_type
{
};

template<class Ntk>
struct has_create_ge<Ntk, std::void_t<decltype( std::declval<Ntk>().create_ge( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_ge_v = has_create_ge<Ntk>::value;
#pragma endregion

#pragma region has_create_xor
template<class Ntk, class = void>
struct has_create_xor : std::false_type
{
};

template<class Ntk>
struct has_create_xor<Ntk, std::void_t<decltype( std::declval<Ntk>().create_xor( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_xor_v = has_create_xor<Ntk>::value;
#pragma endregion

#pragma region has_create_xnor
template<class Ntk, class = void>
struct has_create_xnor : std::false_type
{
};

template<class Ntk>
struct has_create_xnor<Ntk, std::void_t<decltype( std::declval<Ntk>().create_xnor( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_xnor_v = has_create_xnor<Ntk>::value;
#pragma endregion

#pragma region has_create_maj
template<class Ntk, class = void>
struct has_create_maj : std::false_type
{
};

template<class Ntk>
struct has_create_maj<Ntk, std::void_t<decltype( std::declval<Ntk>().create_maj( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_maj_v = has_create_maj<Ntk>::value;
#pragma endregion

#pragma region has_create_ite
template<class Ntk, class = void>
struct has_create_ite : std::false_type
{
};

template<class Ntk>
struct has_create_ite<Ntk, std::void_t<decltype( std::declval<Ntk>().create_ite( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_ite_v = has_create_ite<Ntk>::value;
#pragma endregion

#pragma region has_create_xor3
template<class Ntk, class = void>
struct has_create_xor3 : std::false_type
{
};

template<class Ntk>
struct has_create_xor3<Ntk, std::void_t<decltype( std::declval<Ntk>().create_xor3( std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_xor3_v = has_create_xor3<Ntk>::value;
#pragma endregion

#pragma region has_create_nary_and
template<class Ntk, class = void>
struct has_create_nary_and : std::false_type
{
};

template<class Ntk>
struct has_create_nary_and<Ntk, std::void_t<decltype( std::declval<Ntk>().create_nary_and( std::declval<std::vector<signal<Ntk>>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_nary_and_v = has_create_nary_and<Ntk>::value;
#pragma endregion

#pragma region has_create_nary_or
template<class Ntk, class = void>
struct has_create_nary_or : std::false_type
{
};

template<class Ntk>
struct has_create_nary_or<Ntk, std::void_t<decltype( std::declval<Ntk>().create_nary_or( std::declval<std::vector<signal<Ntk>>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_nary_or_v = has_create_nary_or<Ntk>::value;
#pragma endregion

#pragma region has_create_nary_xor
template<class Ntk, class = void>
struct has_create_nary_xor : std::false_type
{
};

template<class Ntk>
struct has_create_nary_xor<Ntk, std::void_t<decltype( std::declval<Ntk>().create_nary_xor( std::declval<std::vector<signal<Ntk>>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_nary_xor_v = has_create_nary_xor<Ntk>::value;
#pragma endregion

#pragma region has_create_node
template<class Ntk, class = void>
struct has_create_node : std::false_type
{
};

template<class Ntk>
struct has_create_node<Ntk, std::void_t<decltype( std::declval<Ntk>().create_node( std::declval<std::vector<signal<Ntk>>>(), std::declval<kitty::dynamic_truth_table>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_create_node_v = has_create_node<Ntk>::value;
#pragma endregion

#pragma region has_clone_node
template<class Ntk, class = void>
struct has_clone_node : std::false_type
{
};

template<class Ntk>
struct has_clone_node<Ntk, std::void_t<decltype( std::declval<Ntk>().clone_node( std::declval<Ntk>(), std::declval<node<Ntk>>(), std::declval<std::vector<signal<Ntk>>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_clone_node_v = has_clone_node<Ntk>::value;
#pragma endregion

#pragma region has_substitute_node
template<class Ntk, class = void>
struct has_substitute_node : std::false_type
{
};

template<class Ntk>
struct has_substitute_node<Ntk, std::void_t<decltype( std::declval<Ntk>().substitute_node( std::declval<node<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_substitute_node_v = has_substitute_node<Ntk>::value;
#pragma endregion

#pragma region has_substitute_nodes
template<class Ntk, class = void>
struct has_substitute_nodes : std::false_type
{
};

template<class Ntk>
struct has_substitute_nodes<Ntk, std::void_t<decltype( std::declval<Ntk>().substitute_nodes( std::declval<std::list<std::pair<node<Ntk>, signal<Ntk>>>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_substitute_nodes_v = has_substitute_nodes<Ntk>::value;
#pragma endregion

#pragma region has_replace_in_node
template<class Ntk, class = void>
struct has_replace_in_node : std::false_type
{
};

template<class Ntk>
struct has_replace_in_node<Ntk, std::void_t<decltype( std::declval<Ntk>().replace_in_node( std::declval<node<Ntk>>(), std::declval<node<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_replace_in_node_v = has_replace_in_node<Ntk>::value;
#pragma endregion

#pragma region has_replace_in_outputs
template<class Ntk, class = void>
struct has_replace_in_outputs : std::false_type
{
};

template<class Ntk>
struct has_replace_in_outputs<Ntk, std::void_t<decltype( std::declval<Ntk>().replace_in_outputs( std::declval<node<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_replace_in_outputs_v = has_replace_in_outputs<Ntk>::value;
#pragma endregion

#pragma region has_take_out_node
template<class Ntk, class = void>
struct has_take_out_node : std::false_type
{
};

template<class Ntk>
struct has_take_out_node<Ntk, std::void_t<decltype( std::declval<Ntk>().take_out_node( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_take_out_node_v = has_take_out_node<Ntk>::value;
#pragma endregion

#pragma region is_dead
template<class Ntk, class = void>
struct has_is_dead : std::false_type
{
};

template<class Ntk>
struct has_is_dead<Ntk, std::void_t<decltype( std::declval<Ntk>().is_dead( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_dead_v = has_is_dead<Ntk>::value;
#pragma endregion

#pragma region has_substitute_node_of_parents
template<class Ntk, class = void>
struct has_substitute_node_of_parents : std::false_type
{
};

template<class Ntk>
struct has_substitute_node_of_parents<Ntk, std::void_t<decltype( std::declval<Ntk>().substitute_node_of_parents( std::declval<std::vector<node<Ntk>>>(), std::declval<node<Ntk>>(), std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_substitute_node_of_parents_v = has_substitute_node_of_parents<Ntk>::value;
#pragma endregion

#pragma region has_size
template<class Ntk, class = void>
struct has_size : std::false_type
{
};

template<class Ntk>
struct has_size<Ntk, std::void_t<decltype( std::declval<Ntk>().size() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_size_v = has_size<Ntk>::value;
#pragma endregion

#pragma region has_num_cis
template<class Ntk, class = void>
struct has_num_cis : std::false_type
{
};

template<class Ntk>
struct has_num_cis<Ntk, std::void_t<decltype( std::declval<Ntk>().num_cis() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_num_cis_v = has_num_cis<Ntk>::value;
#pragma endregion

#pragma region has_num_cos
template<class Ntk, class = void>
struct has_num_cos : std::false_type
{
};

template<class Ntk>
struct has_num_cos<Ntk, std::void_t<decltype( std::declval<Ntk>().num_cos() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_num_cos_v = has_num_cos<Ntk>::value;
#pragma endregion

#pragma region has_num_latches
template<class Ntk, class = void>
struct has_num_latches : std::false_type
{
};

template<class Ntk>
struct has_num_latches<Ntk, std::void_t<decltype( std::declval<Ntk>().num_latches() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_num_latches_v = has_num_latches<Ntk>::value;
#pragma endregion

#pragma region has_num_pis
template<class Ntk, class = void>
struct has_num_pis : std::false_type
{
};

template<class Ntk>
struct has_num_pis<Ntk, std::void_t<decltype( std::declval<Ntk>().num_pis() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_num_pis_v = has_num_pis<Ntk>::value;
#pragma endregion

#pragma region has_num_pos
template<class Ntk, class = void>
struct has_num_pos : std::false_type
{
};

template<class Ntk>
struct has_num_pos<Ntk, std::void_t<decltype( std::declval<Ntk>().num_pos() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_num_pos_v = has_num_pos<Ntk>::value;
#pragma endregion

#pragma region has_num_gates
template<class Ntk, class = void>
struct has_num_gates : std::false_type
{
};

template<class Ntk>
struct has_num_gates<Ntk, std::void_t<decltype( std::declval<Ntk>().num_gates() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_num_gates_v = has_num_gates<Ntk>::value;
#pragma endregion

#pragma region has_num_registers
template<class Ntk, class = void>
struct has_num_registers : std::false_type
{
};

template<class Ntk>
struct has_num_registers<Ntk, std::void_t<decltype( std::declval<Ntk>().num_registers() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_num_registers_v = has_num_registers<Ntk>::value;
#pragma endregion

#pragma region has_fanin_size
template<class Ntk, class = void>
struct has_fanin_size : std::false_type
{
};

template<class Ntk>
struct has_fanin_size<Ntk, std::void_t<decltype( std::declval<Ntk>().fanin_size( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_fanin_size_v = has_fanin_size<Ntk>::value;
#pragma endregion

#pragma region has_fanout_size
template<class Ntk, class = void>
struct has_fanout_size : std::false_type
{
};

template<class Ntk>
struct has_fanout_size<Ntk, std::void_t<decltype( std::declval<Ntk>().fanout_size( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_fanout_size_v = has_fanout_size<Ntk>::value;
#pragma endregion

#pragma region has_incr_fanout_size
template<class Ntk, class = void>
struct has_incr_fanout_size : std::false_type
{
};

template<class Ntk>
struct has_incr_fanout_size<Ntk, std::void_t<decltype( std::declval<Ntk>().incr_fanout_size( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_incr_fanout_size_v = has_incr_fanout_size<Ntk>::value;
#pragma endregion

#pragma region has_decr_fanout_size
template<class Ntk, class = void>
struct has_decr_fanout_size : std::false_type
{
};

template<class Ntk>
struct has_decr_fanout_size<Ntk, std::void_t<decltype( std::declval<Ntk>().decr_fanout_size( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_decr_fanout_size_v = has_decr_fanout_size<Ntk>::value;
#pragma endregion

#pragma region has_depth
template<class Ntk, class = void>
struct has_depth : std::false_type
{
};

template<class Ntk>
struct has_depth<Ntk, std::void_t<decltype( std::declval<Ntk>().depth() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_depth_v = has_depth<Ntk>::value;
#pragma endregion

#pragma region has_level
template<class Ntk, class = void>
struct has_level : std::false_type
{
};

template<class Ntk>
struct has_level<Ntk, std::void_t<decltype( std::declval<Ntk>().level( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_level_v = has_level<Ntk>::value;
#pragma endregion

#pragma region has_update_levels
template<class Ntk, class = void>
struct has_update_levels : std::false_type
{
};

template<class Ntk>
struct has_update_levels<Ntk, std::void_t<decltype( std::declval<Ntk>().update_levels() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_update_levels_v = has_update_levels<Ntk>::value;
#pragma endregion

#pragma region has_update_mffcs
template<class Ntk, class = void>
struct has_update_mffcs : std::false_type
{
};

template<class Ntk>
struct has_update_mffcs<Ntk, std::void_t<decltype( std::declval<Ntk>().update_mffcs() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_update_mffcs_v = has_update_mffcs<Ntk>::value;
#pragma endregion

#pragma region has_update_topo
template<class Ntk, class = void>
struct has_update_topo : std::false_type
{
};

template<class Ntk>
struct has_update_topo<Ntk, std::void_t<decltype( std::declval<Ntk>().update_topo() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_update_topo_v = has_update_topo<Ntk>::value;
#pragma endregion

#pragma region has_update_fanout
template<class Ntk, class = void>
struct has_update_fanout : std::false_type
{
};

template<class Ntk>
struct has_update_fanout<Ntk, std::void_t<decltype( std::declval<Ntk>().update_fanout() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_update_fanout_v = has_update_fanout<Ntk>::value;
#pragma endregion

#pragma region has_is_on_critical_path
template<class Ntk, class = void>
struct has_is_on_critical_path : std::false_type
{
};

template<class Ntk>
struct has_is_on_critical_path<Ntk, std::void_t<decltype( std::declval<Ntk>().is_on_critical_path( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_on_critical_path_v = has_is_on_critical_path<Ntk>::value;
#pragma endregion

#pragma region has_is_buf
template<class Ntk, class = void>
struct has_is_buf : std::false_type
{
};

template<class Ntk>
struct has_is_buf<Ntk, std::void_t<decltype( std::declval<Ntk>().is_buf( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_buf_v = has_is_buf<Ntk>::value;
#pragma endregion

#pragma region has_is_not
template<class Ntk, class = void>
struct has_is_not : std::false_type
{
};

template<class Ntk>
struct has_is_not<Ntk, std::void_t<decltype( std::declval<Ntk>().is_not( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_not_v = has_is_not<Ntk>::value;
#pragma endregion

#pragma region has_is_and
template<class Ntk, class = void>
struct has_is_and : std::false_type
{
};

template<class Ntk>
struct has_is_and<Ntk, std::void_t<decltype( std::declval<Ntk>().is_and( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_and_v = has_is_and<Ntk>::value;
#pragma endregion

#pragma region has_is_or
template<class Ntk, class = void>
struct has_is_or : std::false_type
{
};

template<class Ntk>
struct has_is_or<Ntk, std::void_t<decltype( std::declval<Ntk>().is_or( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_or_v = has_is_or<Ntk>::value;
#pragma endregion

#pragma region has_is_xor
template<class Ntk, class = void>
struct has_is_xor : std::false_type
{
};

template<class Ntk>
struct has_is_xor<Ntk, std::void_t<decltype( std::declval<Ntk>().is_xor( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_xor_v = has_is_xor<Ntk>::value;
#pragma endregion

#pragma region has_is_maj
template<class Ntk, class = void>
struct has_is_maj : std::false_type
{
};

template<class Ntk>
struct has_is_maj<Ntk, std::void_t<decltype( std::declval<Ntk>().is_maj( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_maj_v = has_is_maj<Ntk>::value;
#pragma endregion

#pragma region has_is_ite
template<class Ntk, class = void>
struct has_is_ite : std::false_type
{
};

template<class Ntk>
struct has_is_ite<Ntk, std::void_t<decltype( std::declval<Ntk>().is_ite( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_ite_v = has_is_ite<Ntk>::value;
#pragma endregion

#pragma region has_is_xor3
template<class Ntk, class = void>
struct has_is_xor3 : std::false_type
{
};

template<class Ntk>
struct has_is_xor3<Ntk, std::void_t<decltype( std::declval<Ntk>().is_xor3( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_xor3_v = has_is_xor3<Ntk>::value;
#pragma endregion

#pragma region has_is_nary_and
template<class Ntk, class = void>
struct has_is_nary_and : std::false_type
{
};

template<class Ntk>
struct has_is_nary_and<Ntk, std::void_t<decltype( std::declval<Ntk>().is_nary_and( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_nary_and_v = has_is_nary_and<Ntk>::value;
#pragma endregion

#pragma region has_is_nary_or
template<class Ntk, class = void>
struct has_is_nary_or : std::false_type
{
};

template<class Ntk>
struct has_is_nary_or<Ntk, std::void_t<decltype( std::declval<Ntk>().is_nary_or( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_nary_or_v = has_is_nary_or<Ntk>::value;
#pragma endregion

#pragma region has_is_nary_xor
template<class Ntk, class = void>
struct has_is_nary_xor : std::false_type
{
};

template<class Ntk>
struct has_is_nary_xor<Ntk, std::void_t<decltype( std::declval<Ntk>().is_nary_xor( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_nary_xor_v = has_is_nary_xor<Ntk>::value;
#pragma endregion

#pragma region has_is_function
template<class Ntk, class = void>
struct has_is_function : std::false_type
{
};

template<class Ntk>
struct has_is_function<Ntk, std::void_t<decltype( std::declval<Ntk>().is_function( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_function_v = has_is_function<Ntk>::value;
#pragma endregion

#pragma region has_get_network_name
template<class Ntk, class = void>
struct has_get_network_name : std::false_type
{
};

template<class Ntk>
struct has_get_network_name<Ntk, std::void_t<decltype( std::declval<Ntk>().get_network_name( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_get_network_name_v = has_get_network_name<Ntk>::value;
#pragma endregion

#pragma region has_set_network_name
template<class Ntk, class = void>
struct has_set_network_name : std::false_type
{
};

template<class Ntk>
struct has_set_network_name<Ntk, std::void_t<decltype( std::declval<Ntk>().set_network_name( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_set_network_name_v = has_set_network_name<Ntk>::value;
#pragma endregion


#pragma region has_node_function
template<class Ntk, class = void>
struct has_node_function : std::false_type
{
};

template<class Ntk>
struct has_node_function<Ntk, std::void_t<decltype( std::declval<Ntk>().node_function( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_node_function_v = has_node_function<Ntk>::value;
#pragma endregion

#pragma region has_get_node
template<class Ntk, class = void>
struct has_get_node : std::false_type
{
};

template<class Ntk>
struct has_get_node<Ntk, std::void_t<decltype( std::declval<Ntk>().get_node( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_get_node_v = has_get_node<Ntk>::value;
#pragma endregion

#pragma region has_make_signal
template<class Ntk, class = void>
struct has_make_signal : std::false_type
{
};

template<class Ntk>
struct has_make_signal<Ntk, std::void_t<decltype( std::declval<Ntk>().make_signal( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_make_signal_v = has_make_signal<Ntk>::value;
#pragma endregion

#pragma region has_is_complemented
template<class Ntk, class = void>
struct has_is_complemented : std::false_type
{
};

template<class Ntk>
struct has_is_complemented<Ntk, std::void_t<decltype( std::declval<Ntk>().is_complemented( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_complemented_v = has_is_complemented<Ntk>::value;
#pragma endregion

#pragma region has_node_to_index
template<class Ntk, class = void>
struct has_node_to_index : std::false_type
{
};

template<class Ntk>
struct has_node_to_index<Ntk, std::void_t<decltype( std::declval<Ntk>().node_to_index( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_node_to_index_v = has_node_to_index<Ntk>::value;
#pragma endregion

#pragma region has_index_to_node
template<class Ntk, class = void>
struct has_index_to_node : std::false_type
{
};

template<class Ntk>
struct has_index_to_node<Ntk, std::void_t<decltype( std::declval<Ntk>().index_to_node( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_index_to_node_v = has_index_to_node<Ntk>::value;
#pragma endregion

#pragma region has_ci_at
template<class Ntk, class = void>
struct has_ci_at : std::false_type
{
};

template<class Ntk>
struct has_ci_at<Ntk, std::void_t<decltype( std::declval<Ntk>().ci_at( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_ci_at_v = has_ci_at<Ntk>::value;
#pragma endregion

#pragma region has_co_at
template<class Ntk, class = void>
struct has_co_at : std::false_type
{
};

template<class Ntk>
struct has_co_at<Ntk, std::void_t<decltype( std::declval<Ntk>().co_at( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_co_at_v = has_co_at<Ntk>::value;
#pragma endregion

#pragma region has_pi_at
template<class Ntk, class = void>
struct has_pi_at : std::false_type
{
};

template<class Ntk>
struct has_pi_at<Ntk, std::void_t<decltype( std::declval<Ntk>().pi_at( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_pi_at_v = has_pi_at<Ntk>::value;
#pragma endregion

#pragma region has_po_at
template<class Ntk, class = void>
struct has_po_at : std::false_type
{
};

template<class Ntk>
struct has_po_at<Ntk, std::void_t<decltype( std::declval<Ntk>().po_at( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_po_at_v = has_po_at<Ntk>::value;
#pragma endregion

#pragma region has_ro_at
template<class Ntk, class = void>
struct has_ro_at : std::false_type
{
};

template<class Ntk>
struct has_ro_at<Ntk, std::void_t<decltype( std::declval<Ntk>().ro_at( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_ro_at_v = has_ro_at<Ntk>::value;
#pragma endregion

#pragma region has_ri_at
template<class Ntk, class = void>
struct has_ri_at : std::false_type
{
};

template<class Ntk>
struct has_ri_at<Ntk, std::void_t<decltype( std::declval<Ntk>().ri_at( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_ri_at_v = has_ri_at<Ntk>::value;
#pragma endregion

#pragma region ci_index
template<class Ntk, class = void>
struct ci_index : std::false_type
{
};

template<class Ntk>
struct ci_index<Ntk, std::void_t<decltype( std::declval<Ntk>().index_to_node( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool ci_index_v = ci_index<Ntk>::value;
#pragma endregion

#pragma region co_index
template<class Ntk, class = void>
struct co_index : std::false_type
{
};

template<class Ntk>
struct co_index<Ntk, std::void_t<decltype( std::declval<Ntk>().index_to_node( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool co_index_v = co_index<Ntk>::value;
#pragma endregion

#pragma region pi_index
template<class Ntk, class = void>
struct pi_index : std::false_type
{
};

template<class Ntk>
struct pi_index<Ntk, std::void_t<decltype( std::declval<Ntk>().index_to_node( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool pi_index_v = pi_index<Ntk>::value;
#pragma endregion

#pragma region po_index
template<class Ntk, class = void>
struct po_index : std::false_type
{
};

template<class Ntk>
struct po_index<Ntk, std::void_t<decltype( std::declval<Ntk>().index_to_node( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool po_index_v = po_index<Ntk>::value;
#pragma endregion

#pragma region ro_index
template<class Ntk, class = void>
struct ro_index : std::false_type
{
};

template<class Ntk>
struct ro_index<Ntk, std::void_t<decltype( std::declval<Ntk>().index_to_node( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool ro_index_v = ro_index<Ntk>::value;
#pragma endregion

#pragma region ri_index
template<class Ntk, class = void>
struct ri_index : std::false_type
{
};

template<class Ntk>
struct ri_index<Ntk, std::void_t<decltype( std::declval<Ntk>().index_to_node( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool ri_index_v = ri_index<Ntk>::value;
#pragma endregion

#pragma region has_ro_to_ri
template<class Ntk, class = void>
struct has_ro_to_ri : std::false_type
{
};

template<class Ntk>
struct has_ro_to_ri<Ntk, std::void_t<decltype( std::declval<Ntk>().ro_to_ri( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_ro_to_ri_v = has_ro_to_ri<Ntk>::value;
#pragma endregion

#pragma region has_ri_to_ro
template<class Ntk, class = void>
struct has_ri_to_ro : std::false_type
{
};

template<class Ntk>
struct has_ri_to_ro<Ntk, std::void_t<decltype( std::declval<Ntk>().ri_to_ro( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_ri_to_ro_v = has_ri_to_ro<Ntk>::value;
#pragma endregion

#pragma region has_foreach_node
template<class Ntk, class = void>
struct has_foreach_node : std::false_type
{
};

template<class Ntk>
struct has_foreach_node<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_node( std::declval<void( node<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_node_v = has_foreach_node<Ntk>::value;
#pragma endregion

#pragma region has_foreach_ci
template<class Ntk, class = void>
struct has_foreach_ci : std::false_type
{
};

template<class Ntk>
struct has_foreach_ci<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_ci( std::declval<void( node<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_ci_v = has_foreach_ci<Ntk>::value;
#pragma endregion

#pragma region has_foreach_co
template<class Ntk, class = void>
struct has_foreach_co : std::false_type
{
};

template<class Ntk>
struct has_foreach_co<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_co( std::declval<void( signal<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_co_v = has_foreach_co<Ntk>::value;
#pragma endregion

#pragma region has_foreach_pi
template<class Ntk, class = void>
struct has_foreach_pi : std::false_type
{
};

template<class Ntk>
struct has_foreach_pi<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_pi( std::declval<void( node<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_pi_v = has_foreach_pi<Ntk>::value;
#pragma endregion

#pragma region has_foreach_po
template<class Ntk, class = void>
struct has_foreach_po : std::false_type
{
};

template<class Ntk>
struct has_foreach_po<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_po( std::declval<void( signal<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_po_v = has_foreach_po<Ntk>::value;
#pragma endregion

#pragma region has_foreach_ro
template<class Ntk, class = void>
struct has_foreach_ro : std::false_type
{
};

template<class Ntk>
struct has_foreach_ro<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_ro( std::declval<void( node<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_ro_v = has_foreach_ro<Ntk>::value;
#pragma endregion

#pragma region has_foreach_ri
template<class Ntk, class = void>
struct has_foreach_ri : std::false_type
{
};

template<class Ntk>
struct has_foreach_ri<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_ri( std::declval<void( signal<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_ri_v = has_foreach_ri<Ntk>::value;
#pragma endregion

#pragma region has_foreach_gate
template<class Ntk, class = void>
struct has_foreach_gate : std::false_type
{
};

template<class Ntk>
struct has_foreach_gate<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_gate( std::declval<void( node<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_gate_v = has_foreach_gate<Ntk>::value;
#pragma endregion

#pragma region has_foreach_register
template<class Ntk, class = void>
struct has_foreach_register : std::false_type
{
};

template<class Ntk>
struct has_foreach_register<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_register( std::declval<void( std::pair<node<Ntk>,signal<Ntk>>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_register_v = has_foreach_register<Ntk>::value;
#pragma endregion

#pragma region has_foreach_fanin
template<class Ntk, class = void>
struct has_foreach_fanin : std::false_type
{
};

template<class Ntk>
struct has_foreach_fanin<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_fanin( std::declval<node<Ntk>>(), std::declval<void( signal<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_fanin_v = has_foreach_fanin<Ntk>::value;
#pragma endregion

#pragma region has_foreach_fanout
template<class Ntk, class = void>
struct has_foreach_fanout : std::false_type
{
};

template<class Ntk>
struct has_foreach_fanout<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_fanout( std::declval<node<Ntk>>(), std::declval<void( node<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_fanout_v = has_foreach_fanout<Ntk>::value;
#pragma endregion

#pragma region has_compute
template<class Ntk, typename T, class = void>
struct has_compute : std::false_type
{
};

template<class Ntk, typename T>
struct has_compute<Ntk, T, std::void_t<decltype( std::declval<Ntk>().compute( std::declval<node<Ntk>>(), std::begin( std::vector<T>() ), std::end( std::vector<T>() ) ) )>> : std::true_type
{
};

template<class Ntk, typename T>
inline constexpr bool has_compute_v = has_compute<Ntk, T>::value;
#pragma endregion

#pragma region has_compute_inplace
template<class Ntk, typename T, class = void>
struct has_compute_inplace : std::false_type
{
};

template<class Ntk, typename T>
struct has_compute_inplace<Ntk, T, std::void_t<decltype( std::declval<Ntk>().compute( std::declval<node<Ntk>>(), std::declval<T&>(), std::begin( std::vector<T>() ), std::end( std::vector<T>() ) ) )>> : std::true_type
{
};

template<class Ntk, typename T>
inline constexpr bool has_compute_inplace_v = has_compute_inplace<Ntk, T>::value;
#pragma endregion

#pragma region has_has_mapping
template<class Ntk, class = void>
struct has_has_mapping : std::false_type
{
};

template<class Ntk>
struct has_has_mapping<Ntk, std::void_t<decltype( std::declval<Ntk>().has_mapping() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_has_mapping_v = has_has_mapping<Ntk>::value;
#pragma endregion

#pragma region has_is_cell_root
template<class Ntk, class = void>
struct has_is_cell_root : std::false_type
{
};

template<class Ntk>
struct has_is_cell_root<Ntk, std::void_t<decltype( std::declval<Ntk>().is_cell_root( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_is_cell_root_v = has_is_cell_root<Ntk>::value;
#pragma endregion

#pragma region has_clear_mapping
template<class Ntk, class = void>
struct has_clear_mapping : std::false_type
{
};

template<class Ntk>
struct has_clear_mapping<Ntk, std::void_t<decltype( std::declval<Ntk>().clear_mapping() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_clear_mapping_v = has_clear_mapping<Ntk>::value;
#pragma endregion

#pragma region has_num_cells
template<class Ntk, class = void>
struct has_num_cells : std::false_type
{
};

template<class Ntk>
struct has_num_cells<Ntk, std::void_t<decltype( std::declval<Ntk>().num_cells() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_num_cells_v = has_num_cells<Ntk>::value;
#pragma endregion

#pragma region has_add_to_mapping
template<class Ntk, class = void>
struct has_add_to_mapping : std::false_type
{
};

template<class Ntk>
struct has_add_to_mapping<Ntk, std::void_t<decltype( std::declval<Ntk>().add_to_mapping( std::declval<node<Ntk>>(), std::begin( std::vector<node<Ntk>>() ), std::end( std::vector<node<Ntk>>() ) ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_add_to_mapping_v = has_add_to_mapping<Ntk>::value;
#pragma endregion

#pragma region has_remove_from_mapping
template<class Ntk, class = void>
struct has_remove_from_mapping : std::false_type
{
};

template<class Ntk>
struct has_remove_from_mapping<Ntk, std::void_t<decltype( std::declval<Ntk>().remove_from_mapping( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_remove_from_mapping_v = has_remove_from_mapping<Ntk>::value;
#pragma endregion

#pragma region has_cell_function
template<class Ntk, class = void>
struct has_cell_function : std::false_type
{
};

template<class Ntk>
struct has_cell_function<Ntk, std::void_t<decltype( std::declval<Ntk>().cell_function( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_cell_function_v = has_cell_function<Ntk>::value;
#pragma endregion

#pragma region has_set_cell_function
template<class Ntk, class = void>
struct has_set_cell_function : std::false_type
{
};

template<class Ntk>
struct has_set_cell_function<Ntk, std::void_t<decltype( std::declval<Ntk>().set_cell_function( std::declval<node<Ntk>>(), std::declval<kitty::dynamic_truth_table>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_set_cell_function_v = has_set_cell_function<Ntk>::value;
#pragma endregion

#pragma region has_foreach_cell_fanin
template<class Ntk, class = void>
struct has_foreach_cell_fanin : std::false_type
{
};

template<class Ntk>
struct has_foreach_cell_fanin<Ntk, std::void_t<decltype( std::declval<Ntk>().foreach_cell_fanin( std::declval<node<Ntk>>(), std::declval<void( node<Ntk>, uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_foreach_cell_fanin_v = has_foreach_cell_fanin<Ntk>::value;
#pragma endregion

#pragma region has_clear_values
template<class Ntk, class = void>
struct has_clear_values : std::false_type
{
};

template<class Ntk>
struct has_clear_values<Ntk, std::void_t<decltype( std::declval<Ntk>().clear_values() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_clear_values_v = has_clear_values<Ntk>::value;
#pragma endregion

#pragma region has_value
template<class Ntk, class = void>
struct has_value : std::false_type
{
};

template<class Ntk>
struct has_value<Ntk, std::void_t<decltype( std::declval<Ntk>().value( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_value_v = has_value<Ntk>::value;
#pragma endregion

#pragma region set_value
template<class Ntk, class = void>
struct has_set_value : std::false_type
{
};

template<class Ntk>
struct has_set_value<Ntk, std::void_t<decltype( std::declval<Ntk>().set_value( std::declval<node<Ntk>>(), uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_set_value_v = has_set_value<Ntk>::value;
#pragma endregion

#pragma region incr_value
template<class Ntk, class = void>
struct has_incr_value : std::false_type
{
};

template<class Ntk>
struct has_incr_value<Ntk, std::void_t<decltype( std::declval<Ntk>().incr_value( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_incr_value_v = has_incr_value<Ntk>::value;
#pragma endregion

#pragma region decr_value
template<class Ntk, class = void>
struct has_decr_value : std::false_type
{
};

template<class Ntk>
struct has_decr_value<Ntk, std::void_t<decltype( std::declval<Ntk>().decr_value( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_decr_value_v = has_decr_value<Ntk>::value;
#pragma endregion

#pragma region has_clear_visited
template<class Ntk, class = void>
struct has_clear_visited : std::false_type
{
};

template<class Ntk>
struct has_clear_visited<Ntk, std::void_t<decltype( std::declval<Ntk>().clear_visited() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_clear_visited_v = has_clear_visited<Ntk>::value;
#pragma endregion

#pragma region has_visited
template<class Ntk, class = void>
struct has_visited : std::false_type
{
};

template<class Ntk>
struct has_visited<Ntk, std::void_t<decltype( std::declval<Ntk>().visited( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_visited_v = has_visited<Ntk>::value;
#pragma endregion

#pragma region set_visited
template<class Ntk, class = void>
struct has_set_visited : std::false_type
{
};

template<class Ntk>
struct has_set_visited<Ntk, std::void_t<decltype( std::declval<Ntk>().set_visited( std::declval<node<Ntk>>(), uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_set_visited_v = has_set_visited<Ntk>::value;
#pragma endregion

#pragma region trav_id
template<class Ntk, class = void>
struct has_trav_id : std::false_type
{
};

template<class Ntk>
struct has_trav_id<Ntk, std::void_t<decltype( std::declval<Ntk>().trav_id() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_trav_id_v = has_trav_id<Ntk>::value;
#pragma endregion

#pragma region incr_trav_id
template<class Ntk, class = void>
struct has_incr_trav_id : std::false_type
{
};

template<class Ntk>
struct has_incr_trav_id<Ntk, std::void_t<decltype( std::declval<Ntk>().incr_trav_id() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_incr_trav_id_v = has_incr_trav_id<Ntk>::value;
#pragma endregion

#pragma region has_get_name
template<class Ntk, class = void>
struct has_get_name : std::false_type
{
};

template<class Ntk>
struct has_get_name<Ntk, std::void_t<decltype( std::declval<Ntk>().get_name( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_get_name_v = has_get_name<Ntk>::value;
#pragma endregion

#pragma region has_set_name
template<class Ntk, class = void>
struct has_set_name : std::false_type
{
};

template<class Ntk>
struct has_set_name<Ntk, std::void_t<decltype( std::declval<Ntk>().set_name( std::declval<signal<Ntk>>(), std::string() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_set_name_v = has_set_name<Ntk>::value;
#pragma endregion

#pragma region has_has_name
template<class Ntk, class = void>
struct has_has_name : std::false_type
{
};

template<class Ntk>
struct has_has_name<Ntk, std::void_t<decltype( std::declval<Ntk>().has_name( std::declval<signal<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_has_name_v = has_has_name<Ntk>::value;
#pragma endregion

#pragma region has_get_output_name
template<class Ntk, class = void>
struct has_get_output_name : std::false_type
{
};

template<class Ntk>
struct has_get_output_name<Ntk, std::void_t<decltype( std::declval<Ntk>().get_output_name( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_get_output_name_v = has_get_output_name<Ntk>::value;
#pragma endregion

#pragma region has_set_output_name
template<class Ntk, class = void>
struct has_set_output_name : std::false_type
{
};

template<class Ntk>
struct has_set_output_name<Ntk, std::void_t<decltype( std::declval<Ntk>().set_output_name( uint32_t(), std::string() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_set_output_name_v = has_set_output_name<Ntk>::value;
#pragma endregion

#pragma region has_has_output_name
template<class Ntk, class = void>
struct has_has_output_name : std::false_type
{
};

template<class Ntk>
struct has_has_output_name<Ntk, std::void_t<decltype( std::declval<Ntk>().has_output_name( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_has_output_name_v = has_has_output_name<Ntk>::value;
#pragma endregion

#pragma region has_new_color
template<class Ntk, class = void>
struct has_new_color : std::false_type
{
};

template<class Ntk>
struct has_new_color<Ntk, std::void_t<decltype( std::declval<Ntk>().new_color() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_new_color_v = has_new_color<Ntk>::value;
#pragma endregion

#pragma region has_current_color
template<class Ntk, class = void>
struct has_current_color : std::false_type
{
};

template<class Ntk>
struct has_current_color<Ntk, std::void_t<decltype( std::declval<Ntk>().current_color() )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_current_color_v = has_current_color<Ntk>::value;
#pragma endregion

#pragma region has_clear_colors
template<class Ntk, class = void>
struct has_clear_colors : std::false_type
{
};

template<class Ntk>
struct has_clear_colors<Ntk, std::void_t<decltype( std::declval<Ntk>().clear_colors( uint32_t() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_clear_colors_v = has_clear_colors<Ntk>::value;
#pragma endregion

#pragma region has_color
template<class Ntk, class = void>
struct has_color : std::false_type
{
};

template<class Ntk>
struct has_color<Ntk, std::void_t<decltype( std::declval<Ntk>().color( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_color_v = has_color<Ntk>::value;
#pragma endregion

#pragma region has_paint
template<class Ntk, class = void>
struct has_paint : std::false_type
{
};

template<class Ntk>
struct has_paint<Ntk, std::void_t<decltype( std::declval<Ntk>().paint( std::declval<node<Ntk>>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_paint_v = has_paint<Ntk>::value;
#pragma endregion

#pragma region has_eval_color
template<class Ntk, class = void>
struct has_eval_color : std::false_type
{
};

template<class Ntk>
struct has_eval_color<Ntk, std::void_t<decltype( std::declval<Ntk>().eval_color( std::declval<node<Ntk>>(), std::declval<void( uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_eval_color_v = has_eval_color<Ntk>::value;
#pragma endregion

#pragma region has_eval_fanins_color
template<class Ntk, class = void>
struct has_eval_fanins_color : std::false_type
{
};

template<class Ntk>
struct has_eval_fanins_color<Ntk, std::void_t<decltype( std::declval<Ntk>().eval_fanins_color( std::declval<node<Ntk>>(), std::declval<void( uint32_t )>() ) )>> : std::true_type
{
};

template<class Ntk>
inline constexpr bool has_eval_fanins_color_v = has_eval_fanins_color<Ntk>::value;
#pragma endregion

/*! \brief SFINAE based on iterator type (for compute functions).
 */
template<typename Iterator, typename T>
using iterates_over_t = std::enable_if_t<std::is_same_v<typename Iterator::value_type, T>, T>;

/*! \brief SFINAE based on iterator type for truth tables (for compute functions).
 */
template<typename Iterator>
using iterates_over_truth_table_t = std::enable_if_t<kitty::is_truth_table<typename Iterator::value_type>::value, typename Iterator::value_type>;

template<class Iterator, typename T>
inline constexpr bool iterates_over_v = std::is_same_v<typename Iterator::value_type, T>;

iFPGA_NAMESPACE_HEADER_END