set (install-decks 
    advection.example.16.in 
    stochsg.512.2d.noout.in
    stochsg.2048.2d.noout.in
    )

install (FILES ${install-decks} DESTINATION decks)
