"""
Provides base parameter structures for SMILES block generation and manipulation in chemical compounds.
This module defines fundamental data structures and constants used throughout the block_smiles
package, including SMARTS patterns for chemical environments, bond definitions, and quality
thresholds for molecular fragments.
Classes:
    BaseParams: Core parameters for SMILES block generation containing SMARTS patterns
        for chemical environments and bond definitions, as well as threshold rules
        for molecular properties.
    AvroSchema: Predefined Avro schema definitions for storing block and combination records,
        with detailed field specifications for serialization.
    ParquetSchema: Predefined Parquet schema tuples for block and combination records,
        compatible with pyarrow for efficient columnar storage.
    RandomizedSmilesThresholds: Manages thresholds for randomized SMILES generation based on
        empirical saturation curves from the MOSES dataset, enabling efficient sampling
        of unique SMILES representations.
"""
from dataclasses import dataclass

# Disclaimer for SMARTS patterns from R-BRICS by Novartis
_DISCLAIMER = """
 Copyright (c) 2009, Novartis Institutes for BioMedical Research Inc.
 All rights reserved.
 Copyright (c) 2022, IBM LLC, Leili Zhang, Vasu Rao, Wendy Cornell
 All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: 

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
    * Neither the name of Novartis Institutes for BioMedical Research Inc. 
      nor the names of its contributors may be used to endorse or promote 
      products derived from this software without specific prior written permission.
    * Neither the name of International Business Machine
      nor the names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Created by Greg Landrum, Nov 2008
Updated by Leili Zhang, Vasu Rao, Jul 2022
 """

@dataclass
class BaseParams:
    """
    Base parameters for smiles block generation
    """
    smarts_disclaimer: str = _DISCLAIMER
    chemical_environs_smarts: tuple[str, str] = (('L1', '[C;D3]([#0,#6,#7,#8])(=O)'),
 ('L10', '[N;R;$(N(@C(=O))@[C,N,O,S])]'),
 ('L11', '[S;D2](-;!@[#0,#6])'),
 ('L12', '[S;D4]([#6,#0])(=O)(=O)'),
 ('L12b', '[S;D4;!R](!@O)(!@O)'),
 ('L13', '[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]'),
 ('L14', '[c;$(c(:[c,n,o,s]):[n,o,s])]'),
 ('L14b', '[RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])]'),
 ('L15', '[C;$(C(-;@C)-;@C)]'),
 ('L16', '[c;$(c(:c):c)]'),
 ('L16b', '[RC;$([RC](@[RC])@[RC])]'),
 ('L17', '[C](-C)(-C)(-C)'),
 ('L18', '[R!#1;x3]'),
 ('L182', '[R!#1;x3]'),
 ('L19', '[R!#1;x2]'),
 ('L192', '[R!#1;x2]'),
 ('L20', '[CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))]'),
 ('L21', '[CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))]'),
 ('L22', '[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))]'),
 ('L23', '[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))]'),
 ('L3', '[O;D2]-;!@[#0,#6,#1]'),
 ('L30', '[C;D2]([#0,#6,#7,#8,#16])(#[N,C])'),
 ('L4', '[C;!D1;!$(C=*)]-;!@[#6]'),
 ('L5', '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]'),
 ('L51', '[N;!R;!D1;$(N(!@[N,O]))]'),
 ('L6', '[C;D3;!R](=O)-;!@[#0,#6,#7,#8]'),
 ('L7a', '[C;D2,D3]-[#6]'),
 ('L7b', '[C;D2,D3]-[#6]'),
 ('L8', '[C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))]'),
 ('L81', '[C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))]'),
 ('L9', '[RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])]'))

    rbond_smarts: tuple[str, str] = (
('L1-L10','[$([C;D3]([#0,#6,#7,#8])(=O))]-[$([N;R;$(N(@C(=O))@[C,N,O,S])])]'),
 ('L1-L3', '[$([C;D3]([#0,#6,#7,#8])(=O))]-[$([O;D2]-;!@[#0,#6,#1])]'),
 ('L1-L5',
  '[$([C;D3]([#0,#6,#7,#8])(=O))]-[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]'),
 ('L10-L13',
  '[$([N;R;$(N(@C(=O))@[C,N,O,S])])]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L10-L14',
  '[$([N;R;$(N(@C(=O))@[C,N,O,S])])]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L10-L14b',
  '[$([N;R;$(N(@C(=O))@[C,N,O,S])])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L10-L15', '[$([N;R;$(N(@C(=O))@[C,N,O,S])])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L10-L16', '[$([N;R;$(N(@C(=O))@[C,N,O,S])])]-[$([c;$(c(:c):c)])]'),
 ('L10-L16b',
  '[$([N;R;$(N(@C(=O))@[C,N,O,S])])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L11-L13', '[$([S;D2](-;!@[#0,#6]))]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L11-L14', '[$([S;D2](-;!@[#0,#6]))]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L11-L14b',
  '[$([S;D2](-;!@[#0,#6]))]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L11-L15', '[$([S;D2](-;!@[#0,#6]))]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L11-L16', '[$([S;D2](-;!@[#0,#6]))]-[$([c;$(c(:c):c)])]'),
 ('L11-L16b', '[$([S;D2](-;!@[#0,#6]))]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L12b-L12b', '[$([S;D4;!R](!@O)(!@O))]-[$([S;D4;!R](!@O)(!@O))]'),
 ('L12b-L13',
  '[$([S;D4;!R](!@O)(!@O))]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L12b-L14', '[$([S;D4;!R](!@O)(!@O))]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L12b-L14b',
  '[$([S;D4;!R](!@O)(!@O))]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L12b-L15', '[$([S;D4;!R](!@O)(!@O))]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L12b-L16', '[$([S;D4;!R](!@O)(!@O))]-[$([c;$(c(:c):c)])]'),
 ('L12b-L16b', '[$([S;D4;!R](!@O)(!@O))]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L12b-L4', '[$([S;D4;!R](!@O)(!@O))]-[$([C;!D1;!$(C=*)]-;!@[#6])]'),
 ('L12b-L5',
  '[$([S;D4;!R](!@O)(!@O))]-[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]'),
 ('L13-L14',
  '[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]-;@,!@;!@[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L13-L14b',
  '[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L13-L15',
  '[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]-;!@;!@[$([C;$(C(-;@C)-;@C)])]'),
 ('L13-L16',
  '[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]-;@,!@;!@[$([c;$(c(:c):c)])]'),
 ('L13-L16b',
  '[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L14-L14',
  '[$([c;$(c(:[c,n,o,s]):[n,o,s])])]-;@,!@;!@[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L14-L14b',
  '[$([c;$(c(:[c,n,o,s]):[n,o,s])])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L14-L15',
  '[$([c;$(c(:[c,n,o,s]):[n,o,s])])]-;@,!@;!@[$([C;$(C(-;@C)-;@C)])]'),
 ('L14-L16', '[$([c;$(c(:[c,n,o,s]):[n,o,s])])]-;@,!@;!@[$([c;$(c(:c):c)])]'),
 ('L14b-L14b',
  '[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L14b-L15',
  '[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L14b-L16',
  '[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]-[$([c;$(c(:c):c)])]'),
 ('L14b-L16b',
  '[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L14b-L17',
  '[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]-[$([C](-C)(-C)(-C))]'),
 ('L15-L16', '[$([C;$(C(-;@C)-;@C)])]-;@,!@;!@[$([c;$(c(:c):c)])]'),
 ('L15-L16b', '[$([C;$(C(-;@C)-;@C)])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L16-L16', '[$([c;$(c(:c):c)])]-;@,!@;!@[$([c;$(c(:c):c)])]'),
 ('L16-L16b', '[$([c;$(c(:c):c)])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L16b-L16b', '[$([RC;$([RC](@[RC])@[RC])])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L17-L10', '[$([C](-C)(-C)(-C))]-[$([N;R;$(N(@C(=O))@[C,N,O,S])])]'),
 ('L17-L11', '[$([C](-C)(-C)(-C))]-[$([S;D2](-;!@[#0,#6]))]'),
 ('L17-L12', '[$([C](-C)(-C)(-C))]-[$([S;D4]([#6,#0])(=O)(=O))]'),
 ('L17-L12b', '[$([C](-C)(-C)(-C))]-[$([S;D4;!R](!@O)(!@O))]'),
 ('L17-L13', '[$([C](-C)(-C)(-C))]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L17-L14', '[$([C](-C)(-C)(-C))]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L17-L15', '[$([C](-C)(-C)(-C))]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L17-L16', '[$([C](-C)(-C)(-C))]-[$([c;$(c(:c):c)])]'),
 ('L17-L16b', '[$([C](-C)(-C)(-C))]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L17-L17', '[$([C](-C)(-C)(-C))]-[$([C](-C)(-C)(-C))]'),
 ('L17-L5',
  '[$([C](-C)(-C)(-C))]-[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]'),
 ('L17-L51', '[$([C](-C)(-C)(-C))]-[$([N;!R;!D1;$(N(!@[N,O]))])]'),
 ('L17-L8',
  '[$([C](-C)(-C)(-C))]-[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]'),
 ('L17-L81', '[$([C](-C)(-C)(-C))]-[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]'),
 ('L17-L9',
  '[$([C](-C)(-C)(-C))]-[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]'),
 ('L18-L19', '[$([R!#1;x3])]-;@;!@[$([R!#1;x2])]'),
 ('L182-L192', '[$([R!#1;x3])]=;@;!@[$([R!#1;x2])]'),
 ('L20-L21',
  '[$([CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))])]-[$([CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))])]'),
 ('L22-L23',
  '[$([CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))])]-'+ \
  '[$([CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))])]'),
 ('L3-L13', '[$([O;D2]-;!@[#0,#6,#1])]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L3-L14', '[$([O;D2]-;!@[#0,#6,#1])]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L3-L14b',
  '[$([O;D2]-;!@[#0,#6,#1])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L3-L15', '[$([O;D2]-;!@[#0,#6,#1])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L3-L16', '[$([O;D2]-;!@[#0,#6,#1])]-[$([c;$(c(:c):c)])]'),
 ('L3-L16b', '[$([O;D2]-;!@[#0,#6,#1])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L3-L4', '[$([O;D2]-;!@[#0,#6,#1])]-[$([C;!D1;!$(C=*)]-;!@[#6])]'),
 ('L30-L10',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([N;R;$(N(@C(=O))@[C,N,O,S])])]'),
 ('L30-L11',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([S;D2](-;!@[#0,#6]))]'),
 ('L30-L12',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([S;D4]([#6,#0])(=O)(=O))]'),
 ('L30-L12b',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([S;D4;!R](!@O)(!@O))]'),
 ('L30-L13',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L30-L14',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L30-L14b',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L30-L15', '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L30-L16', '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([c;$(c(:c):c)])]'),
 ('L30-L16b',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L30-L30',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]'),
 ('L30-L4',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([C;!D1;!$(C=*)]-;!@[#6])]'),
 ('L30-L5',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-'+\
  '[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]'),
 ('L30-L51',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([N;!R;!D1;$(N(!@[N,O]))])]'),
 ('L30-L6',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([C;D3;!R](=O)-;!@[#0,#6,#7,#8])]'),
 ('L30-L81',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]'),
 ('L30-L9',
  '[$([C;D2]([#0,#6,#7,#8,#16])(#[N,C]))]-'\
  '[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]'),
 ('L4-L11', '[$([C;!D1;!$(C=*)]-;!@[#6])]-[$([S;D2](-;!@[#0,#6]))]'),
 ('L4-L5',
  '[$([C;!D1;!$(C=*)]-;!@[#6])]-[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]'),
 ('L5-L12',
  '[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]-[$([S;D4]([#6,#0])(=O)(=O))]'),
 ('L5-L13',
  '[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]-'+ \
  '[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L5-L14',
  '[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]'+ \
  '-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L5-L14b',
  '[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]'+\
  '-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L5-L15',
  '[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L5-L16',
  '[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]-[$([c;$(c(:c):c)])]'),
 ('L5-L16b',
  '[$([N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L51-L1', '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([C;D3]([#0,#6,#7,#8])(=O))]'),
 ('L51-L12', '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([S;D4]([#6,#0])(=O)(=O))]'),
 ('L51-L12b', '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([S;D4;!R](!@O)(!@O))]'),
 ('L51-L13',
  '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L51-L14',
  '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L51-L14b',
  '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L51-L15', '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L51-L16', '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([c;$(c(:c):c)])]'),
 ('L51-L16b', '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L51-L4', '[$([N;!R;!D1;$(N(!@[N,O]))])]-[$([C;!D1;!$(C=*)]-;!@[#6])]'),
 ('L6-L13',
  '[$([C;D3;!R](=O)-;!@[#0,#6,#7,#8])]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L6-L14',
  '[$([C;D3;!R](=O)-;!@[#0,#6,#7,#8])]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L6-L14b',
  '[$([C;D3;!R](=O)-;!@[#0,#6,#7,#8])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L6-L15', '[$([C;D3;!R](=O)-;!@[#0,#6,#7,#8])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L6-L16', '[$([C;D3;!R](=O)-;!@[#0,#6,#7,#8])]-[$([c;$(c(:c):c)])]'),
 ('L6-L16b',
  '[$([C;D3;!R](=O)-;!@[#0,#6,#7,#8])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L7a-L7b', '[$([C;D2,D3]-[#6])]=[$([C;D2,D3]-[#6])]'),
 ('L8-L10',
  '[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]-[$([N;R;$(N(@C(=O))@[C,N,O,S])])]'),
 ('L8-L13',
  '[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L8-L14',
  '[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L8-L14b',
  '[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L8-L15',
  '[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L8-L16',
  '[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]-[$([c;$(c(:c):c)])]'),
 ('L8-L16b',
  '[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L8-L9',
  '[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]'+\
'-[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]'),
 ('L81-L10',
  '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-[$([N;R;$(N(@C(=O))@[C,N,O,S])])]'),
 ('L81-L13',
  '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L81-L14',
  '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L81-L14b',
  '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L81-L15',
  '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L81-L16', '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-[$([c;$(c(:c):c)])]'),
 ('L81-L16b',
  '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-[$([RC;$([RC](@[RC])@[RC])])]'),
 ('L81-L8',
  '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-[$([C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))])]'),
 ('L81-L9',
  '[$([C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))])]-'+\
    '[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]'),
 ('L9-L13',
  '[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]-'+ \
    '[$([C;$(C(-;@[C,N,O,S])-;@[N,O,S])])]'),
 ('L9-L14',
  '[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]-'+ \
    '[$([c;$(c(:[c,n,o,s]):[n,o,s])])]'),
 ('L9-L14b',
  '[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]'+\
    '-[$([RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])])]'),
 ('L9-L15',
  '[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]-[$([C;$(C(-;@C)-;@C)])]'),
 ('L9-L16',
  '[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]-[$([c;$(c(:c):c)])]'),
 ('L9-L16b',
  '[$([RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])])]-'+\
    '[$([RC;$([RC](@[RC])@[RC])])]'))
    threshold_rules_three_extended: tuple[tuple[str, int], ...] = (
        ("MolWt", 300),
        ("nHDonors", 3),
        ("nHAcceptors", 3),
        ("nRotatableBonds", 3),
        ("CrippenlogP", 3),
        ("TPSA", 60)
    )


@dataclass
class AvroSchema:
    """
        AvroSchema class contains predefined Avro schema tuples for block and combination records.
        Attributes:
            block_avro_tuple (tuple): Schema for Avro block 
            combination_avro_tuple (tuple): Schema for viable combinations smiles
    """
    blocks: tuple = (
        ("type", "record"),
        ("name", "BlockQuality"),
        ("doc", """
            Schema for Avro block:
- block: The block of the SMILES string.
- begin_tag: The chemical tag of the first atom in the block.
- end_tag: The chemical tag of the last atom in the block.
- inchi_key: The InChI key of the block.
- MolWt: Molecular weight of the block.
- nHDonors: Number of hydrogen donors.
- nHAcceptors: Number of hydrogen acceptors.
- nRotatableBonds: Number of rotatable bonds.
- CrippenlogP: LogP estimation using the Crippen Algorithm
- TPSA: Topological Polar Surface Area of the block.
- status: Boolean indicating if the block meets the quality thresholds.
"""),
        ("fields", (
            (("name", "block"), ("type", "string")),
            (("name", "begin_tag"), ("type", "string")),
            (("name", "end_tag"), ("type", "string")),
            (("name", "inchi_key"), ("type", "string")),
            (("name", "MolWt"), ("type", "double")),
            (("name", "nHDonors"), ("type", "int")),
            (("name", "nHAcceptors"), ("type", "int")),
            (("name", "nRotatableBonds"), ("type", "int")),
            (("name", "CrippenlogP"), ("type", "double")),
            (("name", "TPSA"), ("type", "double")),
            (("name", "status"), ("type", "boolean"))
        ))
    )
    combinations: tuple = (
        ("type", "record"),
        ("name", "ViableSolution"),
        ("doc", """
Record defining a single viable solution mapping.

Contains the following fields:
- inchkey_encoding (string): InChI key encoding of each block separated by an '_'
- nb_pure_block (int): Number of pure blocks in the molecule
- zinc_id (int): ZINC database identifier for the molecule
"""),
        ("fields", (
            (("name", "inchkey_encoding"), ("type", "string")),
            (("name", "nb_pure_block"), ("type", "int")),
            (("name", "pure"), ("type", "boolean")),
            (("name", "nb_retro_bond"), ("type", "int")),
            (("name", "nb_smiles_bond"), ("type", "int")),
            (("name", "zinc_id"), ("type", "int"))
        ))
    )


@dataclass
class ParquetSchema:
    """ParquetSchema class contains predefined Parquet schema tuples for block and combination 
    records. The tuples follow the format (field_name, field_type) and can be converted to 
    pyarrow schema.
    """
    blocks: tuple = (
        ('block', 'string'),
        ('begin_tag', 'string'),
        ('end_tag', 'string'),
        ('inchi_key', 'string'),
        ('MolWt', 'float64'),
        ('nHDonors', 'int32'),
        ('nHAcceptors', 'int32'),
        ('nRotatableBonds', 'int32'),
        ('CrippenlogP', 'float64'),
        ('TPSA', 'float64'),
        ('status', 'bool')
    )

    combinations: tuple = (
        ('inchkey_encoding', 'string'),
        ('nb_pure_block', 'int32'),
        ('zinc_id', 'int32')
    )

class RandomizedSmilesThresholds:
    """Thresholds for randomized SMILES generation based on empirical saturation curves.
    
    These values represent the number of randomized SMILES needed to achieve 
    specific coverage of the possible unique SMILES for a molecule,
    derived from the MOSES dataset:
    
    - Each threshold corresponds to a quantile (0.1, 0.2, ..., 0.999) of molecules
    - The values indicate how many randomizations are needed to achieve 99% of 
        possible unique SMILES for molecules at that quantile
    - Using these thresholds allows efficient sampling by generating an appropriate 
        number of randomized SMILES based on molecular complexity
    """

    thresholds = (0, 4000, 6000, 8000, 10000, 12000, 16000, 21000,
                27000, 41000, 57000, 102000, 212000)

    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.thresholds) - 1:
            delta = self.thresholds[self.index+1] - self.thresholds[self.index]
            self.index += 1
            return delta
        raise StopIteration
