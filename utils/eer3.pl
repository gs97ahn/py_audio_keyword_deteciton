#!/usr/bin/perl
open(TRUE,"<$ARGV[3]")||die("CANT OPNE TRUE\n");
open(IMPOSTER,"<$ARGV[4]")||die("CANT OPNE IMPOSTER\n");

@targetscores  = <TRUE>;
@nontargetscores = <IMPOSTER>;

close(TRUE);
close(IMPOSTER);

$llimit = $ARGV[0];
$ulimit = $ARGV[1];
$resolution = $ARGV[2];

$eer = 100;
$cost = 10000;
$diff = 100000000;
$bestthreshold = 0;
$threshold = $llimit;
$eermiss = 0;
$eerfa = 0;
$best_diff = 100;

$targetsize = $#targetscores + 1;
$nontargetsize = $#nontargetscores + 1;

while ($threshold<=$ulimit){    

        $miss=0;
        $fa=0;
        foreach $score (@targetscores){
            chomp($score);
                if ($score < $threshold){
		    $miss++;
                }
        }

        foreach $score (@nontargetscores){
	    chomp($score);
                if ($score >= $threshold){
                        $fa++;
                }
        }

	$error_diff = (($miss/$targetsize) - ($fa/$nontargetsize)) * (($miss/$targetsize) - ($fa/$nontargetsize));
        $eer = (($miss/$targetsize) + ($fa/$nontargetsize)) / 2.0;
	
        $miss_rate = $miss / $targetsize;
        $fa_rate = $fa / $nontargetsize;

	print sprintf("%.8f", ${threshold}), ": ", "EER ", sprintf("%.5f", ${eer}), " FRR ", sprintf("%.5f", ${miss_rate}), " FAR ", sprintf("%.5f", ${fa_rate}), " (", ${miss}, "/", ${targetsize}, ") (", ${fa}, "/", ${nontargetsize}, ")", "\n";
        if ($error_diff < $best_diff){
                $bestthreshold = $threshold;
                $best_miss = $miss;
                $best_fa = $fa;
                $best_miss_rate = $miss_rate;
                $best_fa_rate = $fa_rate;
                $best_eer = $eer;
                $best_diff = $error_diff;
        }
	

        $threshold = $threshold + $resolution;
}
$eer = $best_eer * 100;
$acc = 100.0 - ($best_eer * 100);
print "\n";
print sprintf("%.8f", ${bestthreshold}), ": ", "EER ", sprintf("%.5f", ${best_eer}), " FRR ", sprintf("%.5f", ${best_miss_rate}), " FAR ", sprintf("%.5f", ${best_fa_rate}), " (", ${best_miss}, "/", ${targetsize}, ") (", ${best_fa}, "/", ${nontargetsize}, ")", "\n";

