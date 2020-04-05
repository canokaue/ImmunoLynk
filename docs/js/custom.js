/*
Vizion-Al/ML-Digital Marketing Responsive HTML5 Template
Author: iqonicthemes.in
Version: 1.0
Design and Developed by: iqonicthemes.in
*/

/*----------------------------------------------
Index Of Script
------------------------------------------------

1.Page Loader
2.Back To Top
3.Tooltip
4.Accordion
5.Header
6.Magnific Popup
7.Countdown
8.counter
9.Owl Carousel
10.Wow Animation
11.Contact From


------------------------------------------------
Index Of Script
----------------------------------------------*/
(function($) {

    "use strict";
    $(document).ready(function() {

        /*------------------------
        Page Loader
        --------------------------*/
        jQuery("#load").fadeOut();
        jQuery("#loading").delay(0).fadeOut("slow");

        $(".navbar a").on("click", function(event) {
            if (!$(event.target).closest(".nav-item.dropdown").length) {
                $(".navbar-collapse").collapse('hide');
            }
        });

        /*------------------------
        Back To Top
        --------------------------*/
        $('#back-to-top').fadeOut();
        $(window).on("scroll", function() {
            if ($(this).scrollTop() > 250) {
                $('#back-to-top').fadeIn(1400);
            } else {
                $('#back-to-top').fadeOut(400);
            }
        });
        // scroll body to 0px on click
        $('#top').on('click', function() {
            $('top').tooltip('hide');
            $('body,html').animate({
                scrollTop: 0
            }, 800);
            return false;
        });


        /*------------------------
        Tooltip
        --------------------------*/

        $(function() {
            $('[data-toggle="tooltip"]').tooltip()
        });


        /*------------------------
        Accordion
        --------------------------*/
        $('.iq-accordion .iq-ad-block .ad-details').hide();
        $('.iq-accordion .iq-ad-block:first').addClass('ad-active').children().slideDown('slow');
        $('.iq-accordion .iq-ad-block').on("click", function() {
            if ($(this).children('div').is(':hidden')) {
                $('.iq-accordion .iq-ad-block').removeClass('ad-active').children('div').slideUp('slow');
                $(this).toggleClass('ad-active').children('div').slideDown('slow');
            }
        });


        /*------------------------
        Header
        --------------------------*/
        $('.navbar-nav li a').on('click', function(e) {
            var anchor = $(this);
            $('html, body').stop().animate({
                scrollTop: $(anchor.attr('href')).offset().top - 0
            }, 1500);
            e.preventDefault();
        });
        $(window).on('scroll', function() {
            if ($(this).scrollTop() > 100) {
                $('header').addClass('menu-sticky');
            } else {
                $('header').removeClass('menu-sticky');
            }
        });

        /*------------------------
        Magnific Popup
        --------------------------*/

        /*if ($(".popup-gallery").exists()) {*/
        $('.popup-gallery').magnificPopup({
            delegate: 'a.popup-img',
            type: 'image',
            tLoading: 'Loading image #%curr%...',
            mainClass: 'mfp-img-mobile',
            gallery: {
                enabled: true,
                navigateByImgClick: true,
                preload: [0, 1] // Will preload 0 - before current, and 1 after the current image
            },
            image: {
                tError: '<a href="%url%">The image #%curr%</a> could not be loaded.',
                titleSrc: function(item) {
                    return item.el.attr('title') + '<small>by Marsel Van Oosten</small>';
                }
            }
        });
        /*}*/

        /*if ($(".popup-youtube, .popup-vimeo, .popup-gmaps").exists()) {*/
        $('.popup-youtube, .popup-vimeo, .popup-gmaps').magnificPopup({
            disableOn: 700,
            type: 'iframe',
            mainClass: 'mfp-fade',
            removalDelay: 160,
            preloader: false,
            fixedContentPos: false
        });
        /*}*/



        /*------------------------
        Countdown
        --------------------------*/
        $('#countdown').countdown({
            date: '10/01/2019 23:59:59',
            day: 'Day',
            days: 'Days'
        });


        /*------------------------
        counter
        --------------------------*/
        $('.timer').countTo();

        /*------------------------
        Owl Carousel
        --------------------------*/
        $('.owl-carousel').each(function() {
            var $carousel = $(this);
            $carousel.owlCarousel({
                items: $carousel.data("items"),
                loop: $carousel.data("loop"),
                margin: $carousel.data("margin"),
                nav: $carousel.data("nav"),
                dots: $carousel.data("dots"),
                autoplay: $carousel.data("autoplay"),
                autoplayTimeout: $carousel.data("autoplay-timeout"),
                navText: ['<i class="fa fa-angle-left fa-2x"></i>', '<i class="fa fa-angle-right fa-2x"></i>'],
                responsiveClass: true,
                responsive: {
                    // breakpoint from 0 up
                    0: {
                        items: $carousel.data("items-mobile-sm")
                    },
                    // breakpoint from 480 up
                    480: {
                        items: $carousel.data("items-mobile")
                    },
                    // breakpoint from 786 up
                    786: {
                        items: $carousel.data("items-tab")
                    },
                    // breakpoint from 1023 up
                    1023: {
                        items: $carousel.data("items-laptop")
                    },
                    1199: {
                        items: $carousel.data("items")
                    }
                }
            });
        });


        /*------------------------
        Wow Animation
        --------------------------*/
        var wow = new WOW({
            boxClass: 'wow',
            animateClass: 'animated',
            offset: 0,
            mobile: false,
            live: true
        });
        wow.init();


        /*------------------------
        Contact From
        --------------------------*/
        $('#contact').submit(function(e) {
            var form_data=$(this).serialize();
            var flag = 0;
            e.preventDefault(); // Prevent Default Submission
            $('.require').each(function() {
                if ($.trim($(this).val()) == '') {
                    $(this).css("border", "1px solid red");
                    e.preventDefault(); // Prevent Default Submission
                    flag = 1;
                } else {
                    $(this).css("border", "1px solid grey");
                    flag = 0;
                }
            });
            if (grecaptcha.getResponse() == "") {
                flag = 1;
                alert('Please verify Recaptch');

            } else {
                flag = 0;
            }
            if (flag == 0) {
                console.log(form_data);
                $.ajax({            
                        url: 'php/contact-form.php',
                        type: 'POST',
                        data: form_data, // it will serialize the form data
                    })
                    .done(function(data) {
                        console.log("successfully");
                        $("#result").html('Form was successfully submitted.');
                        $('#contact')[0].reset();
                    })
                    .fail(function() {
                        alert('Ajax Submit Failed ...');
                        console.log("fail");

                    });
            }
        });
    });

     /*------------------------
	Cookie
	--------------------------*/
	$(window).load(function() {
		var user = getCookie("digital-marketing");
		if (user == "") {
			$('#cookie_div').css("display", "inherit");
		}
		$('#cookie').on('click', function() {
			checkCookie();
		});
	});

	function setCookie(cname, cvalue) {
		var d = new Date();
		d.setTime(d.getTime() + (24 * 60 * 60 * 1000));
		var expires = "expires=" + d.toGMTString();
		// document.cookie = cname + "=" + cvalue + "," + expires + ", path=/";
		document.cookie = cname + "=" + cvalue + ";" + expires + "; path=/";
		$('#cookie_div').css("display", "none");
	}

	function getCookie(cname) {
		var name = cname + "=";
		var ca = document.cookie.split(';');
		for (var i = 0; i < ca.length; i++) {
			var c = ca[i];
			var cookie_ = c.trim().split('=') || [];
			if (cookie_ != [] && cname == cookie_[0]) {
				return cookie_[1];
			}
		}
		return "";
	}

	function checkCookie() {
		var user = getCookie("digital-marketing");
		if (user == "") {
			$('#cookie_div').css("display", "none");
			setCookie("digital-marketing", "skdfdfdfdfdfgsdf");
		} else {
			$('#cookie_div').css("display", "inherit");
		}
	}

})(jQuery);