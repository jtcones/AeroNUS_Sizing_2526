AIAA: Raymer, D. P., Aircraft Design: A Conceptual Approach, 6th ed., American Institute of Aeronautics and Astronautics, Reston, VA, 2018.
Keane, A. J., Sóbester, A., and Scanlan, J. P., Small Unmanned Fixed-wing Aircraft Design: A Practical Approach, Wiley, Hoboken, NJ, 2017.
Finger, D. F., "Comparative Performance and Benefit Assessment of VTOL and CTOL UAVs," Proceedings of the International Micro Air Vehicle Conference and Flight Competition (IMAV), 2017.
Sztajnbok, I., et al., "Drag Characterization of a Fixed-Wing Unmanned Aerial Vehicle (UAV) with COTS Avionics through Flight Testing," 2025.
Finger, D. F., Bil, C., and Braun, C., "Drag Estimation of Small Fixed-Wing UAVs," The Aeronautical Journal, Vol. 122, No. 1248, 2018.
"Flight Testing Small Electric Powered Unmanned Aerial Vehicles," Technical Report/Paper.
Mattingly, J. D., Heiser, W. H., and Pratt, D. T., Aircraft Engine Design, 2nd ed., American Institute of Aeronautics and Astronautics, Reston, VA, 2002.

https://www.youtube.com/watch?v=geljbqJz1ro - Series on How to design a UAV

https://github.com/leal26/AeroPy#installation - AeroPy Installation

This is the final version of the performance modelling and optimisation code for the purposes of AIAA DBF. 

My objective of this program is to make simple/abstract the complexities of performance modelling and optimisation for AERONUS
So the team can quickly get going for its competition by actually building the plane. Focusing on actual manufacturing.

This project is in python.

I hope to have a base to model the general rc UAV performance. This would not need to have any code written and can be used directly.
Then I want to have an Interface that allows for AIAA competition rules. This requires the team to translate competition requirements into 
the format i have done. 

For optimisation, I ran an experiment between brute force, Genetic Algorithm, Particle Swarm Optimisation.
This would have proved to Dr Tay Chien Ming, Jonathan, our supervisor that the algorithm will find a decent configuration for our plane.

My Final report will go into details of the entire process. 

----------  Aircraft Modelling  --------------------
This process is necessary for us so we can access the performance of the aircraft.

I have decided for each of the 6 components to have their own class. Where their generally fixed parameters will be stored.
The 5 components are wing, tail, fuselage, avionics, propulsion and landing gear.

Goal is that

BASE level is a plane that the team does not need to change.

WRAP level is to add the competition params over so teams can just change it based on competition rules

what structure to adopt.
Each part is a item

Things to look at:
Landing Gear in Plane
Do i really need abstract class? or do i just need dataclasses and to have the relevant fields.
see whether i need the variables in the class more or the methods.