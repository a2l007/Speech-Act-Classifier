#!/usr/local/bin/perl

#########
# usage #
#########
sub usage {
    print "
Usage: use_cm.pl <input file> <classmap file>
    
    use_cm.pl takes the <input file> (single column file of tags) and runs it
    through the <classmap file>, outputting the original and mapped data.

";
    exit(0);
}


$infile=shift(@ARGV);
$cmfile=shift(@ARGV);
if ($cmfile eq "") {
    usage();
    exit(0);
}

open(IN,"$infile") || die "ERROR can't open $infile: $!\n";
open(CM,"$cmfile") || die "ERROR can't open $cmfile: $!\n";

# Read classmap and store in arrays
while ($line=<CM>) {
    ($orig,$new)=split(/\s+/,$line);
    $orig=~s/\*/\.\*/g;             # Adjust wildcard for perl regexp
    $orig=~s/([\|\%\^\-])/\\\1/g;   # Add backslash to certain chars
    push(@orig,$orig);
    push(@new,$new);
}
close(CM);

while (<IN>) {
    chomp;
    $unmapped=$_;
  LOOP:
    for ($j=0;$j<=$#orig;$j++) {
	if (/^$orig[$j]$/) {
	    $_=$new[$j];
	    last LOOP;
	}
    }
    print "$unmapped\t$_\n";
}
