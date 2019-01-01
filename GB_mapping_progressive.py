# File:         GBCG_progressive.py
# Author:       Michael A. Webb
# Description:  This file is used for implementation of a graph-based coarse-graining of molecules
#               using the progressive grouping scheme. Various options are available and can be
#               be observed by executing the script with the -h option. The code requires access
#               to thee numpy, math, copy, os, and datetime modules along with addition modules
#               distributed along with this code. All testing and execution was performed with
#               Python 2.7 as distributed with anaconda.
#Copyright (C) 2019  Michael A. Webb
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

version=1

#==================================================================
# MODULUES
#==================================================================
import sys,argparse
from   math import *
import numpy as np
import time
import aux.lammps_traj_mod as lammpstrj
from   aux.elements_db import *
import copy,os
import datetime

#==================================================================
# CONSTANTS AND DIRECTORIES
#==================================================================
mlight = 3.500
pdbdir= "pdb_files/"
mapdir= "map_files/"
xyzdir= "xyz_files/"
lmpdir= "lammpstrj_files/"

#==================================================================
#  AUX: create_parser
#==================================================================
def create_parser():

  parser = argparse.ArgumentParser(description='Generates a graph-based coarse-grained (GBCG) particle mapping based on a specified degree of connectivity, i.e., the progressive grouping scheme.\
          Details of the formalism can be found in the publication: Michael A. Webb, Jean-Yves Delannoy, and Juan J. de Pablo, "Graph-based approach to Systematic Molecular Coarse-graining," J. Comp. Theory Comput., DOI: 10.1021/acs.jctc.8b00920. See below for various available options.')

  # ARGUMENTS
  parser.add_argument('-data'     ,dest='datafile',default=None,
                      help = 'Name for the lammps data file. This is required for obtaining connectivity information. (default: none)')

  parser.add_argument('-traj',default = None, help = 'Trajectory file(s) with atom coordinates in LAMMPS format. Multiple files can be included in a quoted list with space separation between the file names. (example: "poly.lammpstrj" or "traj.1 traj.2") Note: trajectory files are not actually used in the mapping scheme. Multiple files will be read simultaneously and are intended to for different groups of atoms rather than a sequence for the same group of atoms')

  parser.add_argument('-min_level',dest='min_level',default=2,help = 'Minimum degree level for graph coarsening.\
          After initial reduction, we start at this level and proceed up to the maximum degree level\
          to subsume neighboring nodes. Example: -min_level 2 will start with nodes of connectivity 2, then 3, then ... (default: 2) It is also possible to specify this as a quoted list with as many elements as there are rounds in the coarsening to alter the minimum degree from round to round. Example: -min_level "2 3 2" will set the minimum degree to 2, then 3, then 2 for 3 rounds of coarsening')

  parser.add_argument('-max_level',dest='max_level',default=3,help = 'Maximum degree level for graph coarsening, as explained in the options for -min_level. \
          (default: 3) It is also possible to specify this as a quoated list with as many elements as there are rounds in the coarsening.')

  parser.add_argument('-niter',dest='niter',default=1,help = 'Number of coarsening iterations. (default: 1) ')


  parser.add_argument('-samp_freq'     ,dest='samp_freq',default=1,
                      help = 'Frequency (in terms of frames) to map configurations.\
                              (default = 1)')

  parser.add_argument('-max_samp'     ,dest='max_samp',default=1,
                      help = 'Maximum number of configurations to map.\
                              (default = 1)')

  parser.add_argument('-max_size'     ,dest='max_size',default=None,
                      help = 'Maximum mass of a CG bead. (default = None)')

  parser.add_argument('-similar'     ,dest='simrat',default=1.0,
          help = 'Threshold ratio for evaluating similarity among CG atom types. If the overlap in constituents for two CG atoms types exceeds this number, they will be treated as a single type. (default = 1.0)')

  parser.add_argument('-typing'     ,dest='typing',default="all",
                      help = 'Typing method for CG atom types. Option "all" will use all of the contituent atom types. Option "heavy" will only use the heavy atoms for the CG typing. (default = "all")')


  parser.add_argument('-tmap'     ,dest='tmap',default=None,
                      help = 'File name for listing names for lammps for lammps atom types. If none supplied, then the mass from the data file is used to determine atom names. (default: none)')

  parser.add_argument('-pmap'     ,dest='pmap',default=None,
                      help = 'File name to list initial priority values for lammps atom types. Atom types assigned a value of -1 will be contracted before any coarse-graining iterations. If no file is supplied, then the mass from the data file is used for this purpose, and atoms with mass < -3.5 are given a -1 value. (default: none)')

  return parser

#==================================================================
#  AUX: convert_args
#==================================================================
def convert_args(args):

  # SET FILES
  files = {}
  files['traj'] = [i for i in args.traj.split()]
  files['data'] = args.datafile
  files['names']= args.tmap
  files['pmap'] = args.pmap
  files['summary'] = open("summary.txt","w")

  # SET ARGUMENTS TO OPTIONS STRUCTURE
  options = {}
  options['min_level']   = [int(lvl) for lvl in args.min_level.split()]
  options['max_level']   = [int(lvl) for lvl in args.max_level.split()]
  options['niter']       = int(args.niter)

  # EXECUTE CONSISTENCY CHECKS
  if options['niter'] != len(options['min_level']):
      options['min_level'] = [options['min_level'][0]]*options['niter']
  if options['niter'] != len(options['max_level']):
      options['max_level'] = [options['max_level'][0]]*options['niter']
  options['sfreq']   = int(args.samp_freq)
  options['max_samp']= int(args.max_samp)
  options['sim_ratio']= float(args.simrat)
  options['typing']  = args.typing
  if args.max_size is not None:
      options['max_size'] = float(args.max_size)
  else:
      options['max_size'] = float('inf')

  # PRINT OUT CONFIRMATION OF PARAMETERS
  print_status("Processing input")
  print "# Trajectory file(s): ", files['traj']
  print "# Mapping frequency set to ", repr(options['sfreq']), " frames..."
  print "# Mapping to include up to ", repr(options['max_samp']), " configurations..."
  print "# Minimum degree for coarsening: ", repr(options['min_level']), " ..."
  print "# Maximum degree for coarsening: ", repr(options['max_level']), " ..."
  print "# Number of coarsening rounds set to ", repr(options['niter']), " ..."

  return (files,options)

#==================================================================
#  AUX: print_status
#==================================================================
def print_status(status):
    print("\n********************************************************")
    print("# {}...".format(status))
    print("********************************************************")
    return

#==================================================================
#  AUX: is_number
#==================================================================
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#==================================================================
#  AUX: make_directories
#==================================================================
def make_directories(dirs):
    for diri in dirs:
        if not os.path.exists(diri):
            os.makedirs(diri)
    return

#==================================================================
#  AUX: approximate_sigma
#==================================================================
def approximate_sigma(node,supp,atoms):
    nodeId = atoms['id'][node]
    rmax = atoms['coords'][nodeId].copy()
    rmin = atoms['coords'][nodeId].copy()
    for i in supp:
        iId = atoms['id'][i]
        ri = atoms['coords'][iId].copy()
        rmax[ri>rmax] = ri[ri>rmax]
        rmin[ri<rmin] = ri[ri<rmin]
    rmax += 2.5
    rmin -= 2.5
    dr   = rmax - rmin
    ravg = np.mean(dr)
    return ravg*0.5

#==================================================================
#  AUX: compute_angle
#==================================================================
def compute_angle(ri,rj,rk):
    rji = ri - rj
    rjk = rk - rj
    cost= np.dot(rji,rjk) / np.linalg.norm(rji) / np.linalg.norm(rjk)
    theta = np.arccos(cost) * 180.0 / np.pi
    return theta

#==================================================================
#  AUX: wrap_into_box
#==================================================================
def wrap_into_box(r,box):
    # shift origin to 0 0 0 to do wrapping
    L = np.array([x[1]-x[0] for x in box])
    r -= box[:,0]
    shift = np.array([floor(ri/Li) for ri,Li in zip(r,L)])
    r -= shift*L
    # shift back
    r += box[:,0]
    return r

#==================================================================
#  AUX: open_files
#==================================================================
def open_files(options,mols):
  # make the directories to contain coordinate files
  fxyz    = []
  flmp    = []
  fpdb    = [[] for mol in mols]
  fmap    = []
  fall    = open("atoms.xyz","w")
  fout    = open("summary.txt","w")

  make_directories([pdbdir,xyzdir,lmpdir,mapdir])
  for i, moli in enumerate(mols):
    fname_xyz = xyzdir + "CG.mol_" + repr(i) + ".xyz"
    fname_lmp = lmpdir + "CG.mol_" + repr(i) + ".lammpstrj"
    fname_pdb = pdbdir + "CG.mol_" + repr(i) + ".0.pdb"
    fxyz.append(open(fname_xyz,"w"))
    flmp.append(open(fname_lmp,"w"))
    fpdb[i].append(open(fname_pdb,"w"))
    for iIter in range(options['niter']):
        for lvl in range(options['min_level'][iIter],options['max_level'][iIter]+1,1):
            fname_pdb = pdbdir + "mol_" + repr(i) + "." + repr(iIter+1) + "_" + repr(lvl) + ".pdb"
            fpdb[i].append(open(fname_pdb,"w"))

  fname_map = mapdir + "CG.map"
  fmap.append(open(fname_map,"w"))
  for iIter in range(options['niter']):
        for lvl in range(options['min_level'][iIter],options['max_level'][iIter]+1,1):
          fname_map = mapdir + "iter." + repr(iIter+1) + "_" + repr(lvl) +  ".map"
          fmap.append(open(fname_map,"w"))

  return fxyz,flmp,fpdb,fmap,fall,fout

#==================================================================
# AUX: write_data_file
#==================================================================
def write_data_file(ftyp,atoms,CGmols,box,nOfType,CGmap):

    # acquire system information
    nCgType  = len(nOfType)
    nBonType = 0
    nAngType = 0
    nDihType = 0

    # total number of atoms
    natm = sum([len(i) for i in CGmols])

    # form adjacency matrix and bonds list and get size info
    adjmat = np.zeros([natm+1,natm+1])
    bonds    = []
    bond2typ = []
    btypes = []
    bavg   = [] # stores average bond length and number of occurences per bond type
    id2typ = {}
    coords = {}
    savg   = [[0.,0.] for i in range(nCgType)]
    for mol in CGmols:
        for i in sorted(mol.keys()):
            bead = mol[i]
            radius = approximate_sigma(i,bead['in'],atoms)
            idi = bead['id']
            ityp= bead['type']
            savg[ityp][0] += radius
            savg[ityp][1] += 1.0
            coords[idi] = bead['coords'][:]
            ri  = bead['coords'][:]
            id2typ[idi] = ityp
            for j in bead['adj']:
                idj = mol[j]['id']
                jtyp= mol[j]['type']
                rj  = mol[j]['coords'][:]
                adjmat[idi,idj] = 1
                adjmat[idj,idi] = 1
                if ityp < jtyp:
                    btype = (ityp,jtyp)
                else:
                    btype = (jtyp,ityp)
                if btype not in btypes:
                    btypes.append(btype)
                    bavg.append([0.0,0.0])
                if (idi < idj):
                    bonds.append((idi,idj))
                    dr   = ri - rj
                    bond2typ.append(btypes.index(btype))
                    bavg[btypes.index(btype)][0] += np.linalg.norm(dr)
                    bavg[btypes.index(btype)][1] += 1.0
    nbonds  = len(bonds)

    # angles
    angles  = []
    ang2typ = []
    atypes  = []
    aavg    = []
    for (i,j) in bonds:
        (potkj,) = np.nonzero(adjmat[j,:])
        (potki,) = np.nonzero(adjmat[i,:])
        kjs = [k for k in potkj if k != i]
        kis = [k for k in potki if k != j]
        ityp = id2typ[i]
        jtyp = id2typ[j]
        ri = coords[i][:]
        rj = coords[j][:]

        # check connections from j --> k (i,j,k) or (k,j,i)
        for k in kjs:
            rk = coords[k][:]
            ktyp = id2typ[k]
            if i < k:
                ang = (i,j,k)
            else:
                ang = (k,j,i)
            if ang not in angles:
                if ityp < ktyp:
                    atype = (ityp,jtyp,ktyp)
                else:
                    atype=(ktyp,jtyp,ityp)
                if atype not in atypes:
                    atypes.append(atype)
                    aavg.append([0.,0.])
                theta = compute_angle(ri,rj,rk)
                angles.append(ang)
                ang2typ.append(atypes.index(atype))
                aavg[atypes.index(atype)][0] += theta
                aavg[atypes.index(atype)][1] += 1.0

        # check connections from i --> k (j,i,k) or (k,i,j)
        for k in kis:
            rk = coords[k][:]
            ktyp = id2typ[k]
            if j < k:
                ang = (j,i,k)
            else:
                ang = (k,i,j)
            if ang not in angles:
                if jtyp < ktyp:
                    atype=(jtyp,ityp,ktyp)
                else:
                    atype=(ktyp,ityp,jtyp)
                if atype not in atypes:
                    atypes.append(atype)
                    aavg.append([0.,0.])
                theta = compute_angle(ri,rj,rk)
                angles.append(ang)
                ang2typ.append(atypes.index(atype))
                aavg[atypes.index(atype)][0] += theta
                aavg[atypes.index(atype)][1] += 1.0

    nangles = len(angles)
    deg2 = np.dot(adjmat[:,:],adjmat[:,:])

    # dihedrals
    # XXX right now excludes connections that go to any of the constituent atom angles
    # may need to add impropers
    dihedrals = []
    dih2typ   = []
    dtypes = []
    for (i,j,k) in angles:
        (potlk,) = np.nonzero(adjmat[k,:])
        (potli,) = np.nonzero(adjmat[i,:])
        lks = [l for l in potlk if l not in [i,j]]
        lis = [l for l in potli if l not in [j,k]]
        ityp = id2typ[i]
        jtyp = id2typ[j]
        ktyp = id2typ[k]

        # check (i,j,k,l) or (l,k,j,i)
        for l in lks:
            ltyp = id2typ[l]
            if i < l:
                dih = (i,j,k,l)
            else:
                dih = (l,k,j,i)
            if dih not in dihedrals:
                if ityp < ltyp:
                    dtype=(ityp,jtyp,ktyp,ltyp)
                elif ltyp < ityp:
                    dtype=(ltyp,ktyp,jtyp,ityp)
                else:
                    if jtyp < ktyp:
                        dtype=(ityp,jtyp,ktyp,ltyp)
                    else:
                        dtype=(ltyp,ktyp,jtyp,ityp)
                if dtype not in dtypes:
                    dtypes.append(dtype)
                dihedrals.append(dih)
                dih2typ.append(dtypes.index(dtype))

        # check (l,i,j,k) or (k,j,i,l)
        for l in lis:
            ltyp = id2typ[l]
            if l < k:
                dih = (l,i,j,k)
            else:
                dih = (k,j,i,l)
            if dih not in dihedrals:
                if ltyp < ktyp:
                    dtype=(ltyp,ityp,jtyp,ktyp)
                elif ktyp < ltyp:
                    dtype=(ktyp,jtyp,ityp,ltyp)
                else:
                    if ityp < jtyp:
                        dtype=(ltyp,ityp,jtyp,ktyp)
                    else:
                        dtype=(ktyp,jtyp,ityp,ltyp)
                if dtype not in dtypes:
                    dtypes.append(dtype)
                dihedrals.append(dih)
                dih2typ.append(dtypes.index(dtype))

    deg3 = np.dot(deg2,adjmat)
    ndihedrals = len(dihedrals)

    # write out the data file
    nBonType = len(btypes)
    nAngType = len(atypes)
    nDihType = len(dtypes)
    fid = open("sys.cg.data","w")
    fid.write("LAMMPS data file via GBCG Mapping, version {} {}\n\n".format(version,str(datetime.date.today())))
    fid.write("{} atoms\n".format(natm))
    fid.write("{} atom types\n".format(nCgType))
    fid.write("{} bonds\n".format(nbonds))
    fid.write("{} bond types\n".format(nBonType))
    fid.write("{} angles\n".format(nangles))
    fid.write("{} angle types\n".format(nAngType))
    fid.write("{} dihedrals\n".format(ndihedrals))
    fid.write("{} dihedral types\n\n".format(nDihType))

    # box
    fid.write("{} {} xlo xhi\n".format(box[0,0],box[0,1]))
    fid.write("{} {} ylo yhi\n".format(box[1,0],box[1,1]))
    fid.write("{} {} zlo zhi\n\n".format(box[2,0],box[2,1]))

    # Masses
    fid.write("Masses\n\n")
    for i,CGtype in CGmap.items():
        fid.write("{} {:>10.4f}\n".format(i+1,CGtype['mass'])) # add one from internal typing

    # Pair Coeffs
    fid.write("\nPair Coeffs\n\n")
    for i,CGtype in CGmap.items():
        savg[i][0] /= savg[i][1]
        fid.write("{:<5d} {:>10.4f} {:>10.4f}\n".format(i+1,0.1,savg[i][0]))

    # Bond Coeffs
    fid.write("\nBond Coeffs\n\n")
    ftyp.write("\n~~~~~BOND TYPES~~~~~\n")
    for i,btype in enumerate(btypes):
        bavg[i][0] /= bavg[i][1]
        fid.write("{} {:>8.3f} {:>8.3f}\n".format(i+1,100.0,bavg[i][0]))
        ftyp.write("Type {}: ({})--({})\n".format(i+1,btype[0]+1,btype[1]+1))

    # Angle Coeffs
    ftyp.write("\n~~~~~ANGLE TYPES~~~~~\n")
    fid.write("\nAngle Coeffs\n\n")
    for i,atype in enumerate(atypes):
        aavg[i][0] /= aavg[i][1]
        fid.write("{} {} {:>10.4f}\n".format(i+1,25.,aavg[i][0]))
        ftyp.write("Type {}: ({})--({})--({})\n".format(i+1,atype[0]+1,atype[1]+1,atype[2]+1))

    # Dihedral Coeffs
    fid.write("\nDihedral Coeffs\n\n")
    ftyp.write("\n~~~~~DIHEDRAL TYPES~~~~~\n")
    for i,dtype in enumerate(dtypes):
        fid.write("{} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(i+1,0.0,0.0,0.0,0.0))
        ftyp.write("Type {}: ({})--({})--({})--({})\n".format(i+1,dtype[0]+1,dtype[1]+1,dtype[2]+1,dtype[3]+1))

    # CHECK CHARGES
    qtot = 0.0
    ntot = 0.0
    for j,mol in enumerate(CGmols):
        qmol = 0.0
        for i in sorted(mol.keys()):
            bead = mol[i]
            ityp= bead['type']
            qmol  += CGmap[ityp]['charge']
            qtot  += CGmap[ityp]['charge']
            ntot  += 1.0
        print("# Charge for molecule {}: {}".format(j,qmol))
    qavg = qtot/ntot

    # Atoms
    fid.write("\nAtoms\n\n")
    for j,mol in enumerate(CGmols):
        for i in sorted(mol.keys()):
            bead = mol[i]
            idi = bead['id']
            ityp= bead['type']
            ri  = bead['coords'].copy()
            #ri  = wrap_into_box(ri,box)
            qi  = CGmap[ityp]['charge'] - qavg
            fid.write("{} {} {} {:>8.3f} {:>15.5f} {:>15.5f} {:>15.5f} 0 0 0\n".format(idi,j+1,ityp+1,qi,\
                    bead['coords'][0],bead['coords'][1],bead['coords'][2]))

    # Bonds
    if len(bonds) > 0:
      fid.write("\nBonds\n\n")
      for i,(bond,btype) in enumerate(zip(bonds,bond2typ)):
          fid.write("{} {} {} {}\n".format(i+1,btype+1,bond[0],bond[1]))

    # Angles
    if len(angles) > 0:
      fid.write("\nAngles\n\n")
      for i,(angle,atype) in enumerate(zip(angles,ang2typ)):
          fid.write("{} {} {} {} {}\n".format(i+1,atype+1,angle[0],angle[1],angle[2]))

    # Dihedrals
    if len(dihedrals) > 0:
      fid.write("\nDihedrals\n\n")
      for i,(dihedral,dtype) in enumerate(zip(dihedrals,dih2typ)):
          fid.write("{} {} {} {} {} {}\n".format(i+1,dtype+1,dihedral[0],dihedral[1],dihedral[2],dihedral[3]))

    fid.close()
    ftyp.close()
    return

#==================================================================
# AUX: write_CG_lammpstrj
#==================================================================
def write_CG_lammpstrj(CGatoms,fid,timestep,box):
  N   = len(CGatoms)
  # write the header
  fid.write("ITEM: TIMESTEP\n\t{}\n".format(timestep))
  fid.write("ITEM: NUMBER OF ATOMS\n\t{}\n".format(N))
  fid.write("ITEM: BOX BOUNDS pp pp pp\n")
  fid.write("{:>10.4f}{:>10.4f}\n".format(box[0,0],box[0,1]))
  fid.write("{:>10.4f}{:>10.4f}\n".format(box[1,0],box[1,1]))
  fid.write("{:>10.4f}{:>10.4f}\n".format(box[2,0],box[2,1]))
  fid.write("ITEM: ATOMS id type x y z\n")

  # write the coordinates
  for i in sorted(CGatoms.keys()):
      CGatom = CGatoms[i]
      crds = CGatom['coords'][:].copy()
      crds = wrap_into_box(crds,box)
      typ  = CGatom['type']
      Id   = CGatom['id']
      fid.write("{:<10d}{:<4d}{:>12.4f}{:>12.4f}{:>12.4f}\n".format(Id,typ+1,crds[0],crds[1],crds[2]))
  return


#==================================================================
# AUX: write_CG_map
#==================================================================
def write_CG_map(CGatoms,atoms,map,fid):
  N   = len(CGatoms)
  mass = 0.0
  for i in sorted(CGatoms.keys()):
      fid.write("{:>6d} {:>5d} {:>10.5f} {:>8.3f} {:>6d}".format(\
              CGatoms[i]['id'],CGatoms[i]['type']+1,CGatoms[i]['mass'],CGatoms[i]['charge'],i))
      for j in sorted(CGatoms[i]['in']):
          fid.write(" {:>6d}".format(j))
      fid.write("\n")
  return

#==================================================================
# AUX: write_CG_pdb
#==================================================================
def write_CG_pdb(CGatoms,atoms,map,fid):
  N   = len(CGatoms)
  fid.write("COMPND    UNNAMED\n")
  fid.write("AUTHOR    GENEATED BY CG_mapping.py\n")
  imap = {}
  cnt  = 0
  for i in sorted(CGatoms.keys()):
    cnt += 1
    imap[i] = cnt
    CGatom = CGatoms[i]
    ptr  = atoms['id'][i]
    crds = CGatom['coords'][:]
    typ  = atoms['type'][ptr]
    lbl  = map[typ]
    fid.write("{:<6s}{:>5d} {:<4s}{:1s}{:>3s}  {:>4d}{:<1s}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      {:>4s}{:<s}\n".\
                            format("HETATM",imap[i],lbl[:3]," ","UNL",1," ",\
                                         crds[0],crds[1],crds[2],1.00,0.00,"",lbl[0]))
  for i in sorted(CGatoms.keys()):
    CGatom = CGatoms[i]
    ptr  = atoms['id'][i]
    crds = CGatom['coords'][:]
    typ  = atoms['type'][ptr]
    lbl  = map[typ]
    fid.write("{:<6s}{:>5d}".format("CONECT",imap[i]))
    for cnct in CGatom['adj']:
      fid.write("{:>5d}".format(imap[cnct]))
    fid.write("\n")
  return None

#==================================================================
#  AUX: write_CG_xyz
#==================================================================
def write_CG_xyz(CGatoms,atoms,map,fid):
  N   = len(CGatoms)
  fid.write("{:<} \n\n".format(N))
  for i,CGatom in CGatoms.items():
    ptr  = atoms['id'][i]
    crds = CGatom['coords'][:]
    typ  = atoms['type'][ptr]
    lbl  = map[typ]
    fid.write('{:<10s} {:>15.3f} {:>15.3f} {:>15.3f}\n'.format(lbl,crds[0],crds[1],crds[2]))

#==================================================================
#  AUX: write_xyz
#==================================================================
def write_xyz(atoms,map,fid):
  N   = len(atoms['type'])
  fid.write("{:<} \n\n".format(N))
  for atm,ptr in atoms['id'].items():
    crds = atoms['coords'][ptr][:]
    typ  = atoms['type'][ptr]
    lbl  = map[typ]
    fid.write('{:<10s} {:>15.3f} {:>15.3f} {:>15.3f}\n'.format(lbl,crds[0],crds[1],crds[2]))

#==================================================================
#  AUX: process_mapfile
#==================================================================
def process_mapfile(mapfile,massMap,mapType):
  the_map = {}
  if mapfile is not None:
    fid = open(mapfile,"r")
    lines = [line.strip().split() for line in fid]
    for line in lines:
      if is_number(line[1]):
        the_map[int(line[0])] = float(line[1])
      else:
        the_map[int(line[0])] = line[1]
  else:
    if mapType == "priority":
      # create priority dictionary based on mass
      for i,m in massMap.items():
        if round(m) <= 3.1:
          the_map[i] = -1
        else:
          the_map[i] = 1./round(m)
          #the_map[i] = 1
    elif mapType == "name":
      for i,m in massMap.items():
        the_map[i] = mass2el[m] if m in mass2el else mass2el[min(mass2el.keys(),key=lambda k: abs(k-m))]
  return the_map

#==================================================================
#  AUX: assign_mols
#==================================================================
# determines all molecules as disconnected subgraphs on a graph
def assign_mols(atoms,bonds):
  untested = sorted(atoms['id'].keys())  # add everyone to being untested
  tested   = []                          # initialize list for tracking who has been tested
  queue    = []                          # initialize queue list
  mols     = []
  print("# Total number of atoms to be assigned: {}".format(len(untested)))
  while untested:
    wait  = []                           # initialize wait list
    if not queue:                        # add to queue list if necessary
      queue.append(untested[0])
      mols.append([])
    for i in queue:                      # go through current queue list
      neighbors = bonds[i]               # find neighbor atoms
      mols[-1].append(i)                # add to current molecule
      neighbors = [ni for ni in neighbors if ni not in tested and ni not in queue] # only explore if untested/not in queue
      idi  = atoms['id'][i]
      for j in neighbors:                      # for each neighbor
        idj  = atoms['id'][j]
      tested.append(i)                         # add i to tested listed
      untested.pop(untested.index(i))          # remove i from untested list
      wait.extend(neighbors)                   # add neighbors to wait list
    queue = list(set(wait[:]))

  print("# Total number of molecules: {}".format(len(mols)))
  atoms['nmol'] = len(mols)
  atoms['molid']= [-1]*len(atoms['type'])
  the_mols = [0]*atoms['nmol']
  for i,mol in enumerate(mols):
    print("#---Number of atoms in mol {}: {}".format(i,len(mol)))
    the_mols[i] = sorted(mol)
    for j in mol:
      atoms['molid'][atoms['id'][j]] = i

  print("********************************************************\n\n")

  return atoms,the_mols

#==================================================================
#  AUX: unwrap_mols(fid)
#==================================================================
def unwrap_mols(atoms,bonds,L,halfL):
  # UNWRAP MOLECULES ACCORDING TO PATH WALKS USING BFS
  untested = sorted(atoms['id'].keys())  # add everyone to being untested
  tested   = []                          # initialize list for tracking who has been tested
  queue    = []                          # initialize queue list
  while untested:                        # while there are untested atp,s
    wait  = []                           # initialize wait list
    if not queue:                        # add to queue list if necessary
      queue.append(untested[0])
    for i in queue:                      # go through current queue list
      neighbors = bonds[i]               # find neighbor atoms
      neighbors = [ni for ni in neighbors if ni not in tested and ni not in queue] # only explore if untested
      idi  = atoms['id'][i]
      ri   = atoms['coords'][idi]
      for j in neighbors:                      # for each neighbor
        idj  = atoms['id'][j]
        rj   = atoms['coords'][idj]
        dr   = rj[:] - ri[:]                   # compute distance
        shift= np.array([round(val) for val in dr/L])
        atoms['coords'][idj] -= shift*L  # get nearest image and  adjust coordinates
      tested.append(i)                         # add i to tested listed
      untested.pop(untested.index(i))          # remove i from untested list
      wait.extend(neighbors)                   # add neighbors to wait list
    queue = list(set(wait[:]))                            # make queue the wait list
  return atoms

#==================================================================
#  AUX: get_bead_coords
#==================================================================
def get_bead_coords(chain,atoms):
  nbeads = len(chain['groups'])
  coords = np.zeros([nbeads,3])
  for i,g in enumerate(chain['groups']):   # for each group
    for j in g:				   # for each atom in group
      idj = atoms['id'][j]
      coords[i,:] += atoms['coords'][idj]*atoms['mass'][idj] # add mass-weighted coordinate
    coords[i,:] /= chain['mass'][i]
  return coords

#==================================================================
#  AUX: prioritize
#==================================================================
def prioritize(thisId,avail,atoms,adjlist):
  # first screen by connectivity
  rank = [len(adjlist[i]) for i in avail]
  pval = [atoms['priority'][atoms['id'][i]] for i in avail]
  maxC = max(rank)
  # if there are equal connectivity proceed to priority numbers
  if (rank.count(maxC) > 1 ):
    rank = [pi if ri == maxC else 0 for ri,pi in zip(rank,pval)]
    maxC = max(rank)
    if (rank.count(maxC) > 1):
        printf("# Priority is ambiguous for assignment of atom {} to the following: ".format(thisId))
        for theId,theRank in zip(avail,rank):
            if theRank==maxC:
              printf("{} ".format(theId))
            printf("\n# Consider a different set of priority values. Just rolling with the first one for now...\n")
    return avail[rank.index(maxC)]
  else:
    return avail[rank.index(maxC)]

#==================================================================
#  AUX: get_CG_coords
#==================================================================
def get_CG_coords(mol,atoms):
  for node,group in mol.items():
    nodeId = atoms['id'][node]
    mi     = atoms['mass'][nodeId]
    crds   = mi*atoms['coords'][nodeId][:]
    M      = mi
    for i in group['in']:
      iId  = atoms['id'][i]
      mi   = atoms['mass'][iId]
      crds += mi*atoms['coords'][iId][:]
      M   += mi
    crds /= M
    group['coords'] = crds[:]

  return mol

#==================================================================
#  AUX: add_if_heavy
#==================================================================
def add_if_heavy(node,neighbor,beads,atoms):
    nid=atoms['id'][neighbor]
    mi = atoms['mass'][nid]
    if mi > mlight:
      beads[node]['nheavy'] +=1
    if neighbor in beads:
      for j in beads[neighbor]['in']:
          if j not in beads[node]['in']:
              mj = atoms['mass'][atoms['id'][j]]
              if mj > mlight:
                beads[node]['nheavy'] +=1
    return

#==================================================================
#  AUX: how_many_heavy
#==================================================================
def how_many_heavy(node,nlist,beads,atoms,options):
    nheavy = 0
    for neighbor in nlist:
        nid=atoms['id'][neighbor]
        mi = atoms['mass'][nid]
        if mi > mlight:
          nheavy += 1
        if neighbor in beads:
          for j in beads[neighbor]['in']:
              if j not in beads[node]['in']:
                  mj = atoms['mass'][atoms['id'][j]]
                  if mj > mlight:
                    nheavy +=1
    return nheavy

#==================================================================
#  AUX: get_mass
#==================================================================
def get_mass(atoms,node,group):
    m = atoms['mass'][atoms['id'][node]]
    for i in group:
        m+= atoms['mass'][atoms['id'][i]]
    return m
#==================================================================
#  AUX: contract
#==================================================================
def contract(curr,touched,queue,node,neighbors,neighCN,minCN,atoms,options):
  contractList = [n for n,c in zip(neighbors,neighCN) if c <= minCN] # testing
  mList       =  [ get_mass(atoms,n,curr[n]['in']) for n in contractList]

  # order contract list based on smallest to largest mass
  contractList = [q for (p,q) in sorted(zip(mList,contractList), key=lambda pair: pair[0])]
  mList       =  [ get_mass(atoms,n,curr[n]['in']) for n in contractList]

  # Check how many heavy atoms would be added
  mtot = get_mass(atoms,node,curr[node]['in'])
  addList = []
  for i in contractList:
    madd = get_mass(atoms,i,curr[i]['in'])
    mpot = madd + mtot
    if mpot <= options['max_size']:
        addList.append(i)
        mtot = mpot
  contractList = addList

  #if nheavy_add + curr[node]['nheavy'] > options['max_size']:
  if mtot > options['max_size']:
    touched.add(node)
  else:
    for cnxn in contractList:   # for each
      touched.add(cnxn)         # mark as touched
      curr[node]['in'].add(cnxn) # augment this as contracted to this node
      add_if_heavy(node,cnxn,curr,atoms)
      curr[cnxn]['adj'].remove(node)    # remove contractor from adjacency list of contracted
      curr[node]['adj'].remove(cnxn)         # remove contracted from adjacency list of contractor
      curr[node]['adj'].update(curr[cnxn]['adj'])          # resolve the adjacency list of contracted
      curr[node]['in'].update(curr[cnxn]['in'])           # resolve the container list of contracted

      # modify adjacency lists of neighbors to point to new node
      for i in curr[cnxn]['adj']:
        curr[i]['adj'].remove(cnxn)
        curr[i]['adj'].add(node)
      del curr[cnxn]                          # remove contracted from current list
      while (cnxn in queue):
          queue.pop(queue.index(cnxn))
    touched.add(node)

  return curr,touched,queue

#==================================================================
#  AUX: make_level_queue
#==================================================================
def make_level_queue(beads,lvl,atoms,touched):
  queue = []
  queue = [i for i,beadi in beads.items() if len(beadi['adj']) == lvl ]
  istouched = [1 if i in touched else 0 for i in queue ]
  queue = [i for i in queue if i not in touched]
  nheavy = [beads[node]['nheavy'] for node in queue]
  nn     = [len(beads[node]['adj'] - touched) for node in queue]
  ndg    = [sum([len(beads[cnct]['adj']-touched) for cnct in beads[node]['adj'] - touched]) for node in queue]
  val    = [1000*i + 100*j +10*k for i,j,k in zip(nheavy,nn,ndg)]
  queue  = [q for (p,q) in sorted(zip(val,queue), key=lambda pair: pair[0],reverse=True)]
  return queue

#==================================================================
#  AUX: reorder_queue
#==================================================================
def reorder_queue(queue,touched,beads,tried):
  # ordering 1
  req  = [i for i in queue]

  nheavy = [beads[node]['nheavy'] for node in req]
  nn    = [len(beads[node]['adj']-touched) for node in req]
  ndg   = [sum([len(beads[cnct]['adj']-touched) for cnct in beads[i]['adj'] - touched]) for i in req]
  val    = [1000*i + 100*j +10*k for i,j,k in zip(nheavy,nn,ndg)]
  req = [q for (p,q) in sorted(zip(nn,req), key=lambda pair: pair[0],reverse=True)]
  queue = req
  return queue

#==================================================================
#  AUX: get_overlap
#==================================================================
def get_overlap(listi,listj):
    n = 0
    ni= len(listi)
    nj= len(listj)
    for el in listi:
        if el in listj:
            listj.pop(listj.index(el))
            n += 1
    return float(n)/float(max(ni,nj))

#==================================================================
#  AUX: add_type
#==================================================================
def add_type(options,atoms,Id,group):
    typ    = atoms['type'][Id]
    if options['typing'] == "heavy":
        if atoms['mass'][Id] > mlight:
            group['type'].append(typ)
    else:
        group['type'].append(typ)
    return

#==================================================================
#  AUX: update_charge
#==================================================================
def update_charge(beadList,atoms):
    for node,bead in beadList.items():
        iId = atoms['id'][node]
        q = atoms['charge'][iId]
        for ib in bead['in']:
            jId = atoms['id'][ib]
            q += atoms['charge'][jId]
        bead['charge'] = q
    return

#==================================================================
#  AUX: update_masses
#==================================================================
def update_masses(beadList,atoms):
    for node,bead in beadList.items():
        iId = atoms['id'][node]
        m = atoms['mass'][iId]
        for ib in bead['in']:
            jId = atoms['id'][ib]
            m += atoms['mass'][jId]
        bead['mass'] = m
    return

#==================================================================
#  AUX: temp_types
#==================================================================
def temp_types(options,atoms,beadsList):
    # aggregate and count constituent atom types
    typeList = []

    # iterate over beads and also assign indices
    beadId = 0
    for beads in beadsList:
        for node in sorted(beads.keys()):
            group = beads[node]
            beadId +=1
            group['id'] = beadId
            nodeId = atoms['id'][node]
            add_type(options,atoms,nodeId,group)
            for j in group['in']:
                jId = atoms['id'][j]
                add_type(options,atoms,jId,group)
            theType= [[x,group['type'].count(x)] for x in set(group['type'])]   # count constituent atom types
            theType= [el for el in sorted(theType,key=lambda pair:pair[0])]     # organize in ascending order
            if theType not in typeList:
              typeList.append(theType)
            group['type'] = theType

    # sort the list of possible types for reduction
    nInType = [len(x) for x in typeList]
    typeList= [t for (t,n) in sorted(zip(typeList,nInType), key=lambda pair: pair[1])]

    # Assign all the atom types
    nOfType = [0 for t in typeList]
    for beads in beadsList:
        for node,group in beads.items():
            iType = typeList.index(group['type'])
            group['type']   = iType
            nOfType[iType] += 1

    # Check for similarity in constitution
    uniqueTypes = []
    uniqueExpnd = []
    nunique = 0
    queue   = typeList[:]
    typeMap = {}
    while queue:
        ti = queue.pop()                    # take something out of queue
        listi = []
        # generate the expanded list
        for el in ti:
            listi.extend([el[0]]*el[1])
        # check for similarity to existing unique types
        simScore    = [0]*nunique
        maxSimScore = -1.0
        for j in range(nunique):
            listj = uniqueExpnd[j][:]
            simScore[j] = get_overlap(listi,listj)
        if nunique > 0:
            maxSimScore = max(simScore)
            imax        = simScore.index(maxSimScore)
        if maxSimScore >= options['sim_ratio']:
            typeMap[typeList.index(ti)] = simScore.index(maxSimScore)
        else:
            uniqueTypes.append(ti[:])
            uniqueExpnd.append(listi[:])
            typeMap[typeList.index(ti)] = nunique
            nunique += 1

    # re-assign all the atom types
    nOfType = [0 for i in range(nunique)]
    for beads in beadsList:
        for node,group in beads.items():
            group['type']   = typeMap[group['type']]
            nOfType[group['type']] += 1

    # get average properties for the CG types
    CGmap = {}
    for i in range(nunique):
        CGmap[i] = {'mass': 0.0, 'charge': 0.0}
    for beads in beadsList:
        for i, group in beads.items():
            iCGtype= group['type']
            iId   = atoms['id'][i]
            iType = atoms['type'][iId]
            mi    = atoms['mass'][iId]
            qi    = atoms['charge'][iId]
            CGmap[iCGtype]['mass']   += mi
            CGmap[iCGtype]['charge'] += qi
            for j in group['in']:
                jId = atoms['id'][j]
                jType = atoms['type'][jId]
                mj    = atoms['mass'][jId]
                qj    = atoms['charge'][jId]
                CGmap[iCGtype]['mass']   += mj
                CGmap[iCGtype]['charge'] += qj

    for i,CGtype in CGmap.items():
        CGtype['mass']   /= nOfType[i]
        CGtype['charge'] /= nOfType[i]

    # write out summary
    #print("# Total number of CG beads: {}".format(len(beads)))
    #print("# Total number of CG types: {}".format(nunique))
    #print("               {:^5s} {:^10s} {:^10s}".format("Ncg","<mass>","<charge>"))
    #for i in range(nunique):
    #    print("-CG Type {:>3d}: {:^5d} {:>10.3f} {:>10.3f}".format(i+1,nOfType[i],CGmap[i]['mass'],CGmap[i]['charge']))

    return
#==================================================================
#  AUX: assign_CG_types
#==================================================================
def assign_CG_types(files,options,maps,atoms,beadsList):
    # aggregate and count constituent atom types
    typeList = []

    # iterate over beads and also assign indices
    beadId = 0
    for beads in beadsList:
        for node in sorted(beads.keys()):
            group = beads[node]
            beadId +=1
            group['id'] = beadId
            nodeId = atoms['id'][node]
            add_type(options,atoms,nodeId,group)
            for j in group['in']:
                jId = atoms['id'][j]
                add_type(options,atoms,jId,group)
            theType= [[x,group['type'].count(x)] for x in set(group['type'])]   # count constituent atom types
            theType= [el for el in sorted(theType,key=lambda pair:pair[0])]     # organize in ascending order
            if theType not in typeList:
              typeList.append(theType)
            group['type'] = theType

    # sort the list of possible types for reduction
    nInType = [len(x) for x in typeList]
    typeList= [t for (t,n) in sorted(zip(typeList,nInType), key=lambda pair: pair[1])]

    # Assign all the atom types
    nOfType = [0 for t in typeList]
    for beads in beadsList:
        for node,group in beads.items():
            iType = typeList.index(group['type'])
            group['type']   = iType
            nOfType[iType] += 1

    # Check for similarity in constitution
    uniqueTypes = []
    uniqueExpnd = []
    nunique = 0
    queue   = typeList[:]
    typeMap = {}
    while queue:
        ti = queue.pop()                    # take something out of queue
        listi = []
        # generate the expanded list
        for el in ti:
            listi.extend([el[0]]*el[1])
        # check for similarity to existing unique types
        simScore    = [0]*nunique
        maxSimScore = -1.0
        for j in range(nunique):
            listj = uniqueExpnd[j][:]
            simScore[j] = get_overlap(listi,listj)
        if nunique > 0:
            maxSimScore = max(simScore)
            imax        = simScore.index(maxSimScore)
        if maxSimScore >= options['sim_ratio']:
            typeMap[typeList.index(ti)] = simScore.index(maxSimScore)
        else:
            uniqueTypes.append(ti[:])
            uniqueExpnd.append(listi[:])
            typeMap[typeList.index(ti)] = nunique
            nunique += 1

    # re-assign all the atom types
    nOfType = [0 for i in range(nunique)]
    for beads in beadsList:
        for node,group in beads.items():
            group['type']   = typeMap[group['type']]
            nOfType[group['type']] += 1

    # get average properties for the CG types
    CGmap = {}
    for i in range(nunique):
        CGmap[i] = {'mass': 0.0, 'charge': 0.0}
    for beads in beadsList:
        for i, group in beads.items():
            iCGtype= group['type']
            iId   = atoms['id'][i]
            iType = atoms['type'][iId]
            mi    = atoms['mass'][iId]
            qi    = atoms['charge'][iId]
            CGmap[iCGtype]['mass']   += mi
            CGmap[iCGtype]['charge'] += qi
            for j in group['in']:
                jId = atoms['id'][j]
                jType = atoms['type'][jId]
                mj    = atoms['mass'][jId]
                qj    = atoms['charge'][jId]
                CGmap[iCGtype]['mass']   += mj
                CGmap[iCGtype]['charge'] += qj

    for i,CGtype in CGmap.items():
        CGtype['mass']   /= nOfType[i]
        CGtype['charge'] /= nOfType[i]

    # write out summary
    fid = open("typing.summary.txt","w")
    fid.write("#===========================\n")
    fid.write("# Typing Summary\n")
    fid.write("#===========================\n")
    fid.write("# Total number of CG beads: {}\n".format(len(beads)))
    fid.write("# Total number of CG types: {}\n".format(nunique))
    fid.write("             {:^5s} {:^10s} {:^10s}\n".format("Ncg","<mass>","<charge>"))
    for i in range(nunique):
        fid.write("-CG Type {:>3d}: {:^5d} {:>10.3f} {:>10.3f}\n".format(i+1,nOfType[i],CGmap[i]['mass'],CGmap[i]['charge']))
    fid.write("\n~~~~~CG TYPES~~~~~\n")

    for i in range(nunique):
        fid.write("Type {}: ".format(i+1))
        for j in uniqueExpnd[i]:
            fid.write("{} ".format(j))
        fid.write("\n")

    return fid,CGmap,nOfType

#==================================================================
#  AUX: write_groups
#==================================================================
def write_groups(i,beads,atoms,map):
  fid = open("mol_" + repr(i) + ".groups.dat","w")
  for node,group in beads.items():
      nhvy = group['nheavy']
      nall = len(group['in']) + 1
      ptr  = atoms['id'][node]
      typ  = atoms['type'][ptr]
      lbl  = map[typ]
      fid.write("{}({}) {} {}-- ".format(node,lbl,nall,nhvy))
      for neighbor in group['adj']:
          ptr = atoms['id'][neighbor]
          typ = atoms['type'][ptr]
          lbl = map[typ]
          fid.write("{}({}) ".format(neighbor,lbl))
      fid.write("\n\t >> {} : ".format(len(group['in'])))
      for atom in group['in']:
          ptr = atoms['id'][atom]
          typ = atoms['type'][ptr]
          lbl = map[typ]
          fid.write("{}({}) ".format(atom,lbl))
      fid.write("\n")
  fid.close()
  return

#==================================================================
#  AUX: redution_mapping
#==================================================================
# determines the grouping of atoms that form CG beads. Information is in
# a dictionary with the following fields
# 'adj' - the adjacency list (set of atoms for bonding)
# 'in'  - constituent atoms
# 'nheavy' - number of heavy atoms in the CG atom
# 'type' - type of the CG bead
# 'coords' - coordinates of the bead
# 'id' - assigned ID
def reduction_mapping(files,options,moli,atoms,adjlist):
  history = []
  curr = {}
  queue= []

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Set up initial list and mark atoms to be reduced
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  files['summary'].write("Initial Number of atoms: {}\n".format(len(moli)))
  print("# Beginning coarse-graining of molecule with {} atoms".format(len(moli)))
  for i in moli:
    idi     = atoms['id'][i] # key to index in rest of atoms structure
    if (atoms['priority'][idi] == -1):
      queue.append(i)
    else:
        mi = atoms['mass'][idi]
        qi = atoms['charge'][idi]
        if mi > mlight:
            curr[i] = {'adj' : set(adjlist[i]),'in': set(), 'nheavy':1, 'type': [],'mass': mi,'charge': qi}
        else:
            curr[i] = {'adj' : set(adjlist[i]),'in': set(), 'nheavy':0, 'type': [],'mass': mi,'charge': qi}
  print("# Initial contraction of {} atoms into {} remaining groups"\
          .format(len(queue),len(curr)))

  # Perform initial contraction for negative priority atoms
  files['summary'].write("Initial contraction consists of {} into {} groups\n".format(len(queue),len(curr)))
  for i in queue:
    neighbors = adjlist[i][:]                           # determine the set of available neighbors
    mergeId   = prioritize(i,neighbors,atoms,adjlist)   # find who to contract into
    neighbors.pop(neighbors.index(mergeId))             # remove contractor from the available neighbors of contracted
    curr[mergeId]['in'].add(i)                          # augment list to reflect the contraction
    add_if_heavy(mergeId,i,curr,atoms)
    curr[mergeId]['adj'].remove(i)                      # remove the contracted from adjacency list of contractor
    curr[mergeId]['adj'].update(neighbors)              # resolve the adjacency list of contracted
  update_masses(curr,atoms)
  update_charge(curr,atoms)
  history.append(copy.deepcopy(curr))

  # Start coordination level reductions, contracting from degree 2 and up
  for iIter in range(options['niter']):
    print_status("Iteration {}".format(iIter+1))
    files['summary'].write("Reduction Round {}\n".format(iIter+1))
    files['summary'].write("Initial number of groups: {}\n".format(len(curr)))
    touched = set()
    for lvl in range(options['min_level'][iIter],options['max_level'][iIter]+1):
      print("# Examining nodes with degree equal to {}...".format(lvl))
      queue = make_level_queue(curr,lvl,atoms,touched)
      tried = set()
      print("# There are {} nodes in the queue...".format(len(queue)))
      while queue:
        node      = queue.pop(0)  # obtain index for first in queue
        if get_mass(atoms,node,curr[node]['in']) >= options['max_size']:
            touched.add(node)
        else:
            neighbors = curr[node]['adj'] - touched  # set of neighbors who are not in the touch list
            if neighbors:
              neighCN   = [len(curr[n]['adj']) for n in neighbors] # number of connections for neighbor (including with 'node')
              minCN = min(neighCN)
              if (minCN == lvl):            # if no one with lower connectivity,
                if (node in tried):
                  curr,touched,queue = contract(curr,touched,queue,node,neighbors,neighCN,minCN,atoms,options)
                else:
                  tried.add(node)
                  queue.append(node)        # then send to end of queue for now
              elif (minCN > lvl):           # if only higher connectivity
                touched.add(node)
              else:                         # otherwise find all lowest
                minCN = lvl - 1 # testing
                curr,touched,queue = contract(curr,touched,queue,node,neighbors,neighCN,minCN,atoms,options)

            else:
              touched.add(node)
            queue = reorder_queue(queue,touched,curr,tried)
      update_masses(curr,atoms)
      update_charge(curr,atoms)
      print("# Queue is exhausted and vertex groups formed...")
      print("# There are currently {} vertex groups...".format(len(curr)))
      files['summary'].write("Reduction at level {} --> {} groups\n".format(lvl,len(curr)))
      history.append(copy.deepcopy(curr))

  files['summary'].write("\n\n")
  return curr,history

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  SUB-MAIN: CGmapping
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def CGmapping(files,options):

  # OPEN THE FILES AND INITIALIZE KEY VARIABLES
  frame      = 0
  nsamp      = 0
  ntraj      = len(files['traj'])  # number of trajectory files
  ftraj      = [0]*ntraj           # list with trajectory file ids
  still_data = [0]*ntraj           # list to indicate if data remains to be read
  tan        = 0.0

  # SCREEN FILE AND GENERATE MAPS
  print_status("Screening files")
  maps              = {}
  atoms             = lammpstrj.screen_frame(files['traj'],"all")
  maps['mass']      = lammpstrj.get_mass_map(files,atoms)
  maps['names']     = process_mapfile(files['names'],maps['mass'],"name")
  maps['priority']  = process_mapfile(files['pmap'],maps['mass'],"priority")
  atoms['mass']     = [maps['mass'][typ] for typ in atoms['type']]
  atoms['priority'] = [maps['priority'][typ] for typ in atoms['type']]
  bonds         = lammpstrj.get_adj_list(files,atoms)
  atoms,mols = assign_mols(atoms,bonds)

  # PERFORM THE MAPPING FOR EACH MOLECULE
  print_status("Performing mapping")
  CGmol   = []
  CGhist  = []
  fxyz,flmp,fpdb,fmap,fall,fout = open_files(options,mols)
  for i,moli in enumerate(mols):
    files['summary'].write("Reduction Summary for molecule {}\n\n".format(i))
    CGmoli,histi = reduction_mapping(files,options,moli,atoms,copy.deepcopy(bonds))
    write_groups(i,CGmoli,atoms,maps['names'])
    CGmol.append(CGmoli)
    CGhist.append(histi)

  # ASSIGN CG ATOM TYPES
  print_status("Assigning preliminary CG site types")
  ftyp,maps['CGtypes'],nCgType = assign_CG_types(files,options,maps,atoms,CGmol)

  # GET TYPES AT EACH HISTORY LEVEL
  nhist = len(CGhist[0])
  nmol  = len(CGhist)
  tmpCGmol  = [[] for i in range(nhist)]
  for i in range(nhist):
      for j in range(nmol):
          tmpCGmol[i].extend([CGhist[j][i]])
  for i in range(nhist):
      temp_types(options,atoms,tmpCGmol[i])

  # BEGIN PROCESSING, FRAME BY FRAME
  print_status("Mapping supplied trajectories")
  for i,f in enumerate(files['traj']):
    ftraj[i]      = open(f,"r")
    still_data[i] = ftraj[i].readline()

  # READ DATA WHILE ALL FILES STILL OPEN
  while all(still_data) and nsamp < options['max_samp']:
    # IF NEED TO SAMPLE
    if (frame%options['sfreq'] == 0):
      ti = time.time()
      # PROCESS TRAJECTORY FRAME
      (atoms,L,halfL,box) = lammpstrj.process_frame(ftraj,"all",atoms)

      # UNWRAP MOLECULES TO COMPUTE ANY SELF PROPERTIES
      atoms           = unwrap_mols(atoms,bonds,L,halfL)
      write_xyz(atoms,maps['names'],fall)

      # WRITE OUT COORDINATES FOR THE GROUPS
      molcpy = []
      for i,CGmoli in enumerate(CGmol):
        for j,histj in enumerate(CGhist[i]):
          if nsamp == 0:
            tmp = get_CG_coords(copy.deepcopy(histj),atoms)
            write_CG_pdb(tmp,atoms,maps['names'],fpdb[i][j])
            write_CG_map(tmp,atoms,maps['names'],fmap[j])
        tmp = get_CG_coords(copy.deepcopy(CGmoli),atoms)
        molcpy.append(tmp)
        write_CG_lammpstrj(tmp,flmp[i],nsamp,box)
        write_CG_xyz(tmp,atoms,maps['names'],fxyz[i])

      # WRITE OUT DATA FILE
      if (nsamp == 0):
        write_data_file(ftyp,atoms,molcpy,box,nCgType,maps['CGtypes'])

      # FINISH BOOK KEEPING
      tan   += time.time()-ti
      nsamp += 1

    # OTHERWISE... SKIP FRAME
    else:
      for fid in ftraj:
        lammpstrj.skip_frame(fid)
    frame+=1
    if (frame%1000==0): print "# ", repr(nsamp), " samples taken after ", repr(frame), " frames..."
    # GET CONTINUE CONDITION
    for i,fid in enumerate(ftraj):
      still_data[i] = fid.readline()

  print_status("Wrapping up...")
  tan /= nsamp


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  MAIN: _main
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def main(argv):
  # PARSE ARGUMENTS
  parser = create_parser()
  args                         = parser.parse_args()
  (files,options)              = convert_args(args)

  # BEGIN ANALYSIS
  t0 = time.time()
  tavg = CGmapping(files,options)

  print("# Elapsed time: {} seconds".format(time.time()-t0))

#==================================================================
#  RUN PROGRAM
#==================================================================
if __name__ == "__main__":
  main(sys.argv[1:])
