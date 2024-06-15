set (install-decks 
  advection.example.16.pin.in
  blast_wave_3d.512.pin.in
  blast_wave_3d.1024.pin.in
  blast_wave_3d.2048.pin.in
  blast_wave_3d.4096.pin.in
  stochsg_2d.512.pin.in
  stochsg_2d.1024.pin.in
  stochsg_2d.2048.pin.in
)

install (FILES ${install-decks} DESTINATION decks)
