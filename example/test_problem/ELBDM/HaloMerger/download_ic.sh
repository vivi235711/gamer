filename=UM_IC_halo_m22_1_Mh_4e9
link=http://use.yt/upload/8812b7bf


curl ${link} -o ${filename}.tar.gz
tar -zxvf ${filename}.tar.gz
rm ${filename}.tar.gz

ln -s ${filename} UM_IC_Halo1
ln -s ${filename} UM_IC_Halo2
