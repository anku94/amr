PREFIX=/l0/amr-umbrella/build
INSTDIR=/users/ankushj/repos/amr-workspace/amr-psmhack-prefix/lib

PSM=$PREFIX/psm-prefix/src/psm
cd $PSM
make -j

tar cf - libpsm_infinipath.so* | ( cd $INSTDIR && tar xf - )

cd ipath
tar cf - libinfinipath.so* | ( cd $INSTDIR && tar xf - )

MVAPICH=$PREFIX/mvapich-prefix/src/mvapich
cd $MVAPICH
make install -j
