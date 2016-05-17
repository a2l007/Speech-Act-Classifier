#!/usr/local/bin/perl
#
# Take Hot Spot Annotations and parse them, mapping them appropriately to
# other lines

# Strip unneeded tags so we can categorize the hot spot more easily
sub strip_tags {
    my $temp = shift @_;

    $temp=~s/-//g;
    $temp=~s/^r\.|\.r$//g;
    $temp=~s/\.r\./\./g;
    return $temp;
}


################
# MAIN PROGRAM #
################
$mrdb_root=$ENV{'MRDB_ROOT'};

@meetings=`ls ../data/*.dadb`;
while ($meet=shift @meetings) {
    print $meet;
    chomp $meet;
    open(HS,"$meet") || die "ERROR can't open file: $!\n";

    @lines=<HS>;

    $prev_beg=9999999;

    # FIRST PASS: Go through lines backwards to record when next hot spot starts
    foreach $line (reverse @lines) {
	@hs=split(/,/,$line);
	$id="$hs[0],$hs[1],$hs[6]";

	# Make non-labeled portions explicit
	$hs[11]=~s/\*\*$/\*\*0/;
	$hs[11]=~s/^\*\*/0\*\*/;

	# Split hot spot into parts to process
	@hs_parts=split(/\*\*|\#\#/,$hs[11]);
	@hs_sep=split(/[^\*\#]+/,$hs[11]);
	shift @hs_sep;
	$i=0;
	$loop=0;
	foreach $part (@hs_parts) {
	    $loop=1;
	    # Parse hot spot part
	    $part=~/^((\d+)=([lwh])([-0+]?)=([acdgos]+(;[acdgos]+)*))?\.?([anbrce]?-?(.[anbrce]-?)*)$/;
	    ($index,$level,$degree,$types,$tags)=($2,$3,$4,$5,$7);

	    # Strip unneeded info for determining category
	    $temp=&strip_tags($tags);
	    
	    # Record start time when beginning of hot spot
	    if ($temp=~/^a$|^a\.b$|^b\.a$|^n\.b$|^b\.n$/) {
		$prev_beg=$hs[0];
		$prev_index=$index;
	    }
	    $beg{$id}=$prev_beg;
	    $index{$id}=$prev_index;
	}

	$beg{$id}=$prev_beg unless ($loop);
	$index{$id}=$prev_index unless ($loop);

    }

    $in_hs=0;
    $end=0;

    foreach $hs (@lines) {
	$cat=$index=$level=$degree=$types=$tags="";
	$comp_index=$comp_level=$comp_degree=$comp_types=$comp_tags="";
	chomp $hs;
	@hs=split(/,/,$hs);

	$id="$hs[0],$hs[1],$hs[6]";
	print "$hs[2]\t";

	# Make non-labeled portions explicit
	$hs[11]=~s/\*\*$/\*\*0/;
	$hs[11]=~s/^\*\*/0\*\*/;

	# Split hot spot into parts to process
	@hs_parts=split(/\*\*|\#\#/,$hs[11]);
	@hs_sep=split(/[^\*\#]+/,$hs[11]);
	shift @hs_sep;
	$i=0;
	foreach $part (@hs_parts) {

	    # Parse hot spot part
	    $part=~/^((\d+)=([lwh])([-0+]?)=([acdgos]+(;[acdgos]+)*))?\.?([anbrce]?-?(.[anbrce]-?)*)$/;
	    ($index,$level,$degree,$types,$tags)=($2,$3,$4,$5,$7);

	    # Strip unneeded info for determining category
	    $temp=&strip_tags($tags);
	    
	    # Inherit proper info 
	    if ($temp=~/^n\.b\.e$|^a\.b\.e$/) {
		$cat.="SING".$hs_sep[$i];
		$in_hs=0;
	    } elsif ($temp=~/^a$|^a\.b$|^b\.a$|^n\.b$|^b\.n$/) {
		$cat.="TRIG".$hs_sep[$i];
		$in_hs=1;

		# For cases with ** where trigger doesn't have other labels
		$index=$prev_index unless ($index);
		$level=$prev_level unless ($level);
		$degree=$prev_degree unless ($degree);
		$types=$prev_types unless ($types);

	    } elsif ($temp=~/^c$|^b\.c$|^c\.b$|^b\.e$|^e\.b$/) {
		$cat.="CLOS".$hs_sep[$i];
		$index=$prev_index;
		$level=$prev_level;
		$degree=$prev_degree;
		$types=$prev_types;
		$in_hs=0;
		
		# Note end time of the ending hotspot
		$end=$hs[1];
	    } elsif ($in_hs) {
		$cat.="INTR".$hs_sep[$i];
		$index=$prev_index;
		$level=$prev_level;
		$degree=$prev_degree;
		$types=$prev_types;
	    } elsif (!$in_hs) {
		if ($hs[0]<$end) {
		    if ($hs[1]<$end) {
			$cat.="INTR($prev_index)".$hs_sep[$i];
		    } else {
			$cat.="B_END($prev_index)".$hs_sep[$i];
		    }

		    # For cases with **
		    $index=$prev_index unless ($index);
		    $level=$prev_level unless ($level);
		    $degree=$prev_degree unless ($degree);
		    $types=$prev_types unless ($types);

		} elsif ($hs[1]>$beg{$id}) {
		    $cat.="B_BEG($index{$id})".$hs_sep[$i];
		} else {
		    $cat.="NONE".$hs_sep[$i];
		}
	    }
	    $prev_index=$index;
	    $prev_level=$level;
	    $prev_degree=$degree;
	    $prev_types=$types;

	    $comp_index.=$index.$hs_sep[$i];
	    $comp_level.=$level.$hs_sep[$i];
	    $comp_degree.=$degree.$hs_sep[$i];
	    $comp_types.=$types.$hs_sep[$i];
	    $comp_tags.=$tags.$hs_sep[$i];

	    $i++;
	}


	# Deal with when there is no hot spot label
	unless ($cat) {
	    if ($in_hs) {
		$cat="INTR";
		$comp_index=$prev_index;
		$comp_level=$prev_level;
		$comp_degree=$prev_degree;
		$comp_types=$prev_types;
	    } else {
		if ($hs[0]<$end && $hs[1]>$beg{$id}) {
		    if ($hs[1]<$end) {
			$cat="INTR($prev_index)+B_BEG($index{$id})";
		    } else {
			$cat="B_END($prev_index)+B_BEG($index{$id})";
		    }
		} elsif ($hs[0]<$end) {
		    if ($hs[1]<$end) {
			$cat="INTR($prev_index)";
		    } else {
			$cat="B_END($prev_index)";
		    }
		} elsif ($hs[1]>$beg{$id}) {
		    $cat="B_BEG($index{$id})";
		} else {
		    $cat="NONE";
		}
	    }
	}

	printf("%15s | %16s | %6s | %4s | %4s | %8s | %6s\n",$hs[11],$cat,$comp_index,$comp_level,$comp_degree,$comp_types,$comp_tags,);
    }
    close(HS);
}
