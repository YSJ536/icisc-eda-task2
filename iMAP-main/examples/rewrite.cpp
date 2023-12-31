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

#include "io/reader.hpp"
#include "io/writer.hpp"
#include "io/detail/write_verilog.hpp"
#include "optimization/rewrite.hpp"
#include "database/views/depth_view.hpp"
#include "database/network/aig_network.hpp"
#include "database/network/klut_network.hpp"
#include "utils/mem_usage.hpp"
#include "algorithms/debugger.hpp"
#include "utils/tic_toc.hpp"
#include <cstdlib>

int main(int argc, char **argv) {
    std::string file = std::string(argv[1]);

    iFPGA_NAMESPACE::write_verilog_params ports;
    iFPGA_NAMESPACE::aig_network aig;
    iFPGA_NAMESPACE::Reader reader(file, aig, ports);
    iFPGA_NAMESPACE::aig_network raig(aig);

    printf("Stats of AIG: pis=%d, pos=%d, area=%d\n", aig.num_pis(), aig.num_pos(), aig.num_gates());

    iFPGA_NAMESPACE::rewrite_params ps;
    raig = iFPGA_NAMESPACE::rewrite(raig, ps);

    printf("Stats of AIG: pis=%d, pos=%d, area=%d\n", aig.num_pis(), aig.num_pos(), aig.num_gates());
    printf("Stats of AIG: pis=%d, pos=%d, area=%d\n", raig.num_pis(), raig.num_pos(), raig.num_gates());
    return 0;
}
